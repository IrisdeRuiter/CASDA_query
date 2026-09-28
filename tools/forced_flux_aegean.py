"""Forced (priorized) flux extraction using AegeanTools.

This is a replacement for the aperture-sum approach in `tools.flux`
(`measure_flux_density`, inspired by Martin Hardcastle's `radioflux.py`).
Instead of summing pixels within a fixed radius, it uses AegeanTools'
priorized fitting to fit a PSF-convolved elliptical Gaussian at a *fixed*
sky position, giving a proper integrated flux density and uncertainty from
the fit covariance rather than a pixel sum.

See: https://github.com/PaulHancock/Aegean/wiki/Priorized-Fitting

Usage from the main notebook, in place of `flux.measure_flux_density`:

    from tools import forced_flux_aegean

    int_flux, err_int_flux, local_rms = forced_flux_aegean.measure_flux_density(
        fitsimage, ra, dec, bmaj, bmin, bpa
    )

or as a drop-in for the notebook's own `force_extract_flux` helper:

    force_flux_int, force_flux_int_err, local_rms = forced_flux_aegean.force_extract_flux(
        fits_image_filename, data, ra, dec, bmaj
    )
"""

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

from AegeanTools.source_finder import SourceFinder
from AegeanTools.models import ComponentSource
from AegeanTools.wcs_helpers import Beam


def _build_prior_source(ra, dec, bmaj, bmin, bpa, peak_flux_guess):
    """Build a single `ComponentSource` to seed the priorized fit.

    `bmaj`/`bmin` are given in degrees (as in a FITS header) and converted
    to the arcsec expected by AegeanTools' `a`/`b` shape parameters.
    """

    src = ComponentSource()
    src.ra = ra
    src.dec = dec
    src.peak_flux = peak_flux_guess
    src.a = bmaj * 3600.
    src.b = bmin * 3600.
    src.pa = bpa
    # Mark the catalog psf as identical to the image psf (in arcsec), so
    # that AegeanTools' resizing step is a no-op (ratio=1) instead of
    # raising on the otherwise-NaN default psf_a/b.
    src.psf_a = src.a
    src.psf_b = src.b
    src.psf_pa = bpa

    return src


def measure_flux_density(
    fitsimage,
    ra,
    dec,
    bmaj,
    bmin=None,
    bpa=0.0,
    rms=None,
    stage=1,
    cores=1,
):
    """Force-fit the flux density at a fixed sky position with AegeanTools.

    Parameters
    ----------
    fitsimage : str
        Path to the FITS image (or cutout) to measure.
    ra, dec : float
        Sky position (decimal degrees) at which to force the fit.
    bmaj, bmin : float
        Restoring beam major/minor axis (degrees). Used both as the fixed
        source shape for the fit, and (via `beam=`) as the image's
        synthesized beam, overriding/filling in for the FITS header.
        If `bmin` is not given, `bmaj` is used for both axes (circular beam).
    bpa : float
        Beam position angle (degrees East of North). Default 0.
    rms : float, optional
        A fixed rms (image units, e.g. Jy/beam) to use for the whole image.
        If None, AegeanTools estimates the background/rms internally (BANE),
        which needs a reasonably large image to work well; for small
        cutouts, pass a pre-computed rms instead (see `force_extract_flux`).
    stage : int
        Priorized fitting stage: 1 = flux only (position and shape held
        fixed at the given values -- true forced photometry), 2 = flux +
        position, 3 = flux + position + shape. Default 1.
    cores : int
        Number of CPU cores used by AegeanTools. Default 1 (safe for use
        inside a Jupyter notebook).

    Returns
    -------
    (int_flux, err_int_flux, local_rms) : tuple of float
        Integrated flux density, its uncertainty, and the local rms, all in
        the image's flux units (e.g. Jy/beam). NaN triple if no fit could
        be made (e.g. the position is not covered by the image).
    """

    if bmin is None:
        bmin = bmaj

    with fits.open(fitsimage) as hdul:
        data = np.squeeze(hdul[0].data)

    finite = data[np.isfinite(data)]
    peak_guess = np.nanmax(np.abs(finite)) if finite.size else 1e-3

    prior = _build_prior_source(ra, dec, bmaj, bmin, bpa, peak_guess)

    sf = SourceFinder()
    sources = sf.priorized_fit_islands(
        filename=fitsimage,
        catalogue=[prior],
        stage=stage,
        rms=rms,
        beam=Beam(bmaj, bmin, bpa),
        doregroup=False,
        progress=False,
        cores=cores,
    )

    if len(sources) == 0:
        return np.nan, np.nan, np.nan

    source = sources[0]

    return float(source.int_flux), float(source.err_int_flux), float(source.local_rms)


def force_extract_flux(fits_image_filename, data, ra, dec, bmaj, bmin=None, bpa=0.0, stage=1, cores=1):
    """Drop-in replacement for the notebook's ad-hoc `force_extract_flux`.

    Same call signature and return convention (integrated flux density and
    its error in mJy, plus the local rms in mJy/beam), but the fit itself
    is done with AegeanTools' priorized fitting instead of an aperture sum.

    - fits_image_filename: location of the fits image on which to perform
      the forced flux extraction.
    - data: grid of pixel values in the fits image, used for noise
      estimation (as in the original helper).
    - ra, dec: location for flux extraction.
    - bmaj, bmin, bpa: beam shape (bmaj/bmin in degrees, bpa in degrees).
      If bmin is not given, bmaj is used for both axes.
    """

    data_flat = data.ravel()
    data_flat_not_nan = data_flat[~np.isnan(data_flat)]

    if not ((len(data_flat_not_nan) > (0.6 * len(data_flat))) and (len(data_flat_not_nan) > 200)):
        return np.nan, np.nan, np.nan

    filtered_data = sigma_clip(data_flat_not_nan, sigma=3)
    local_rms = np.std(filtered_data)

    int_flux, err_int_flux, _ = measure_flux_density(
        fits_image_filename,
        ra,
        dec,
        bmaj,
        bmin=bmin,
        bpa=bpa,
        rms=local_rms,
        stage=stage,
        cores=cores,
    )

    if np.isnan(int_flux):
        return np.nan, np.nan, local_rms * 1000.

    return int_flux * 1000., err_int_flux * 1000., local_rms * 1000.
