# Download data


import os
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import unique

# TAP client for CASDA VO tools
from astroquery.utils.tap.core import TapPlus

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def set_batch_size_workers(nof_downloads):
    batch_size=10      # how many rows you send to `casda.cutout` per request
    max_workers=10      # how many batches to run in parallel (threads)
    if nof_downloads<=20:
        batch_size = 5
        max_workers = 4
    if nof_downloads>20 and nof_downloads<100 :
        batch_size = 20
        max_workers = 5
    if nof_downloads>100:
        batch_size = 20
        max_workers = 10

    return batch_size, max_workers

def prep_urls_cutouts_parallel(
    ra, dec, casda, stokes,
    max_retries=2,      # retries for a failing batch before giving up
    backoff_base=2.0,   # exponential backoff base: sleep = base**attempt + small_jitter
):
    """
    Create cutout URLs from CASDA for a given sky position and Stokes parameter,
    running multiple `casda.cutout` calls in parallel by batching the input table.

    Notes
    -----
    - This function returns the **list of URLs** (checksum links filtered out).
      It does not download files; you can feed the returned list into a parallel
      downloader afterwards.
    - Concurrency is applied at the *batch* level to avoid too many small remote
      calls and to keep per-call overhead reasonable.
    - Retries with exponential backoff help with transient network/service hiccups.

    Parameters
    ----------
    ra, dec : float
        Source coordinates in degrees (ICRS).
    stokes : str
        'I' or 'V'.
    max_retries : int, optional
        Maximum number of retry attempts per batch on failure before skipping it.
    backoff_base : float, optional
        Base for exponential backoff between retries. Sleep is computed as
        (backoff_base ** attempt) + uniform(0, 0.5).

    Returns
    -------
    list of str
        Flattened list of cutout URLs (excluding '.checksum' entries).
    """

    # Coordinates and region of interest
    coord = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
    radius = 5 * u.arcmin

    # -------------------------------------------------------------------------
    # 1) Query CASDA metadata around the target and filter to desired products
    # -------------------------------------------------------------------------
    result_table = casda.query_region(coordinates=coord, radius=radius)
    public_data = casda.filter_out_unreleased(result_table)

    # Build a boolean mask for the desired science products
    mask = (
        ((public_data['quality_level'] == 'GOOD') | (public_data['quality_level'] == 'UNCERTAIN')) &
        (public_data['dataproduct_subtype'] == 'cont.restored.t0') &
        (public_data['pol_states'] == f'/{stokes}/')
    )
    to_cutout = public_data[mask]

    # Deduplicate by obs_id so we don't request redundant cutouts
    to_cutout_unique = unique(to_cutout, keys='obs_id')

    n = len(to_cutout_unique)
    print(f'There are {n} cutouts to be generated for Stokes {stokes}.')
    if n == 0:
        return []

    batch_size, max_workers = set_batch_size_workers(n)
    print(f'Found {n} unique cutouts, preparing batches of {batch_size} (max_workers={max_workers}).')

    # -------------------------------------------------------------------------
    # 2) Create disjoint batches (tuples of (start_index, table_slice))
    # -------------------------------------------------------------------------
    batches = [(i, to_cutout_unique[i:i + batch_size]) for i in range(0, n, batch_size)]
    total_batches = len(batches)

    # -------------------------------------------------------------------------
    # 3) Define a worker that runs casda.cutout for a single batch with retries
    # -------------------------------------------------------------------------
    def _run_batch(idx, tbl_slice):
        """
        Run `casda.cutout` on a slice of the filtered table.

        Returns
        -------
        (int, list[str], Exception|None)
            The slice start index (for ordering later), the list of URLs
            (no checksum links), and an exception (None if success).
        """
        attempts = 0

        # Prepare a compact batch identifier for logging
        ids = list(tbl_slice['obs_id'])
        

        while True:
            try:
                # Perform the remote cutout call for this batch
                url_list = casda.cutout(tbl_slice, coordinates=coord, radius=radius)

                # Filter out checksum helper files
                url_list = [u for u in url_list if not u.endswith('.checksum')]

                print(f'Batch {idx // batch_size + 1}/{total_batches}: '
                      f'cutout returned {len(url_list)} URLs for obs_ids={ids}')
                return idx, url_list, None

            except Exception as e:
                attempts += 1
                print(f'Batch {idx // batch_size + 1}/{total_batches} failed (attempt {attempts}) '
                      f'obs_ids={ids}: {e}')
                traceback.print_exc()

                # Give up after max_retries attempts
                if attempts >= max_retries:
                    print(f'Batch {idx // batch_size + 1}/{total_batches}: '
                          f'giving up after {max_retries} attempts.')
                    return idx, [], e

                # Exponential backoff + small jitter to avoid lockstep retry storms
                sleep_s = (backoff_base ** attempts) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

    # -------------------------------------------------------------------------
    # 4) Execute batches concurrently with a thread pool
    # -------------------------------------------------------------------------
    url_lists_by_idx = {}  # holds results keyed by batch start index so we can re-order
    errors = 0

    if total_batches == 1:
        # Avoid thread overhead for a single batch
        idx, urls, err = _run_batch(batches[0][0], batches[0][1])
        url_lists_by_idx[idx] = urls
        errors += 1 if err else 0
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_run_batch, idx, tbl): idx for (idx, tbl) in batches}
            for fut in as_completed(futures):
                idx, urls, err = fut.result()
                url_lists_by_idx[idx] = urls
                errors += 1 if err else 0

    # -------------------------------------------------------------------------
    # 5) Flatten URLs preserving original batch order
    # -------------------------------------------------------------------------
    url_list_flat = []
    for idx in sorted(url_lists_by_idx.keys()):
        url_list_flat.extend(url_lists_by_idx[idx])

    print(f'All batches done. Total URLs: {len(url_list_flat)}; batches with errors: {errors}')


    return url_list_flat


def get_img_query_obs_ids(
    ra, dec, casda,
    ):
    """
    Get unique obsids from image query. This can be used to check that we have all the catalogues.
    """

    stokes = 'I'
    # Coordinates and region of interest
    coord = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
    radius = 5 * u.arcmin

    # -------------------------------------------------------------------------
    # 1) Query CASDA metadata around the target and filter to desired products
    # -------------------------------------------------------------------------
    result_table = casda.query_region(coordinates=coord, radius=radius)
    public_data = casda.filter_out_unreleased(result_table)

    # Build a boolean mask for the desired science products
    mask = (
        ((public_data['quality_level'] == 'GOOD') | (public_data['quality_level'] == 'UNCERTAIN')) &
        (public_data['dataproduct_subtype'] == 'cont.restored.t0') &
        (public_data['pol_states'] == f'/{stokes}/')
    )
    to_cutout = public_data[mask]

    # Deduplicate by obs_id so we don't request redundant cutouts
    to_cutout_unique = unique(to_cutout, keys='obs_id')

    return list(to_cutout_unique['obs_id'])



def prep_urls_catalogues_region(
    ra, dec, casda,
    max_retries=2,       # retries per batch
    backoff_base=2.0,    # backoff base; sleep = base**attempt + small_jitter
):
    """
    Find Selavy continuum *component* catalogues around (ra, dec) via TAP, then
    stage them from CASDA in parallel batches and return the list of URLs.

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees.
    radius_arcmin : float, optional
        Search radius used when querying obscore (default 5 arcmin).
    max_retries : int, optional
        Max retry attempts per batch on failure.
    backoff_base : float, optional
        Exponential backoff base between retries.

    Returns
    -------
    list[str]
        Flattened list of staged file URLs ('.checksum' removed).
    """

    # -----------------------------
    # 1) Build and execute TAP ADQL
    # -----------------------------
    # Use an ADQL spatial predicate against s_region. We keep it simple and
    # rely on CONTAINS with a POINT at (ra, dec). You can adapt to CIRCLE if preferred.
    tap_qry = (
        "SELECT * FROM ivoa.obscore "
        "WHERE dataproduct_subtype = 'catalogue.continuum.component' "
        f"AND 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), s_region)"
    )

    tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
    job = tap.launch_job_async(tap_qry)
    data = job.get_results()

    # Filter out unreleased rows (CASDA helper)
    cat_data = casda.filter_out_unreleased(data)

    # Optionally reduce duplicates so we don't re-stage the same observation
    # (obscore can have multiple rows per obs_id). Adjust 'keys' as needed
    # e.g., 'obs_publisher_did' if you want product-level uniqueness.
    try:
        cat_data_unique = unique(cat_data, keys='obs_id')
    except Exception:
        # Fall back to original if obs_id missing; you can choose another key here
        cat_data_unique = cat_data

    n = len(cat_data_unique)
    print(f'>>> Found {n} catalogue entries to create urls for after querying on coordinates.')
    if n == 0:
        return []
    batch_size, max_workers = set_batch_size_workers(n)

    # -----------------------------
    # 2) Make batches
    # -----------------------------
    batches = [(i, cat_data_unique[i:i + batch_size]) for i in range(0, n, batch_size)]
    total_batches = len(batches)
    print(f'>>> Preparing {total_batches} batches of size ≤ {batch_size} (max_workers={max_workers}).')

    # -----------------------------
    # 3) Worker with retries
    # -----------------------------
    def _run_batch(idx, tbl_slice):
        """
        Stage a batch via casda.stage_data with retries.

        Returns
        -------
        (int, list[str], Exception|None)
            Batch start index, URL list (checksum filtered), error (None if OK).
        """
        attempts = 0
        # For logging, try to extract some IDs
        try:
            ids = list(tbl_slice['obs_id'])
        except Exception:
            ids = [f'<no obs_id> len={len(tbl_slice)}']

        while True:
            try:
                # Stage the selected catalogue products for this batch
                url_list = casda.stage_data(tbl_slice)
                # Remove checksum helper files
                url_list = [u for u in url_list if not u.endswith('.checksum')]

                print(f'>>> Batch {idx // batch_size + 1}/{total_batches}: '
                      f'staged {len(url_list)} file(s) for obs_ids={ids}')
                return idx, url_list, None

            except Exception as e:
                attempts += 1
                print(f'>>> Batch {idx // batch_size + 1}/{total_batches} FAILED (attempt {attempts}) '
                      f'obs_ids={ids}: {e}')
                traceback.print_exc()

                if attempts >= max_retries:
                    print(f'>>> Batch {idx // batch_size + 1}/{total_batches}: '
                          f'giving up after {max_retries} attempts.')
                    return idx, [], e

                # Exponential backoff with jitter
                sleep_s = (backoff_base ** attempts) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

    # -----------------------------
    # 4) Execute in parallel
    # -----------------------------
    url_lists_by_idx = {}
    errors = 0

    if total_batches == 1:
        idx, urls, err = _run_batch(batches[0][0], batches[0][1])
        url_lists_by_idx[idx] = urls
        errors += 1 if err else 0
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_run_batch, idx, tbl): idx for (idx, tbl) in batches}
            for fut in as_completed(futures):
                idx, urls, err = fut.result()
                url_lists_by_idx[idx] = urls
                errors += 1 if err else 0

    # -----------------------------
    # 5) Flatten in original order
    # -----------------------------
    url_list_flat = []
    for idx in sorted(url_lists_by_idx.keys()):
        url_list_flat.extend(url_lists_by_idx[idx])

    print(f'>>> All catalogue batches done. Total URLs: {len(url_list_flat)}; '
          f'batches with errors: {errors}')

    return url_list_flat, list(data['obs_id'])



def prep_urls_catalogues_obsid(obsid_list, casda):
    """For some reason, not all catalogues are captured by the region query.
    Download remaining catalogs based on their obsid."""
    
    url_list_total = []
    for obsid in obsid_list:
        tap_qry=("SELECT * FROM ivoa.obscore "
                                       "where( (dataproduct_subtype = 'catalogue.continuum.component')"
                                         f"and (obs_id = '{obsid}'))")
                                         #f"and (1=CONTAINS(POINT('ICRS', {ra}, {dec}), s_region)) )")
    
        tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
        job = tap.launch_job_async(tap_qry)
        data = job.get_results()
        
        # filter out the unreleased data
        cat_data = casda.filter_out_unreleased(data)
        url_list = casda.stage_data(cat_data)
        
        url_list = [url for url in url_list if url.endswith('.checksum')==False]
        url_list_total.extend(url_list)
    
    return url_list_total


def prep_urls_catalogues_parallel(ra, dec, casda):
    """Prep url list with catalog data.
    First call the function that queries based on coordinates.
    This doesn't seem to cathc all catalogues. Add an additionaly query based on the obsids
    that were retrieved in the cutout search.
    Combine the resulting url lists."""

    url_list1, obs_ids_cat_from_region = prep_urls_catalogues_region(ra, dec, casda)

    obs_id_list_images = get_img_query_obs_ids(ra, dec, casda)
    
    remaining_download_list = []
    for obs in obs_id_list_images:
        if obs not in obs_ids_cat_from_region:
            remaining_download_list.append(obs)
    
    print(f'>>> For {len(remaining_download_list)} observations no catalogue was identified based on the source region.')
    print('>>> I will retry for these observations by querying on the obsid identifier of the cutouts.')

    url_list2 = prep_urls_catalogues_obsid(remaining_download_list, casda)
    return url_list1 + url_list2