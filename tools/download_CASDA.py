import os
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


def _sanitize_url(url):
    """Escape characters that CASDA's presigned S3 URLs sometimes leave
    unencoded but that break the request in practice:
    - '+' must be %2B in the path for S3 SigV4 signature verification
      (a literal '+' is valid URI syntax so requests/urllib3 won't touch
      it, but it doesn't match what the server signed).
    - raw spaces/quotes (seen in a malformed
      'response-content-disposition=attachment; filename ="..."' query
      param) are illegal in a URL and can trip a 400 from an intermediate
      proxy.
    Already-percent-encoded sequences like the %2B/%20/%22 this produces
    survive untouched through requests, so no further "don't re-encode
    this" flag is needed (unlike yarl/aiohttp, which needs encoded=True).
    """
    return (
        url.replace("+", "%2B")
        .replace(" ", "%20")
        .replace('"', "%22")
    )


def _error_detail(e):
    """Build a diagnostic message for a download failure, including the
    server's response body when available (e.g. the S3 <Code>/<Message>
    XML on a 403, which is far more informative than the generic
    'Forbidden' text requests shows by default)."""
    detail = str(e)
    response = getattr(e, 'response', None)
    if response is not None:
        try:
            body = response.text.strip()
            if body:
                detail = f'{detail} | server response: {body[:500]}'
        except Exception:
            pass
    return detail


def download_urls_parallel(
    urls,
    casda,
    savedir,
    max_workers=6,
    max_retries=3,
    backoff=2.0
):
    """
    Download a list of URLs in parallel using casda.download_files().
    Each URL is downloaded individually so failures do not affect others.

    Returns
    -------
    list of str
        The URLs that still failed after exhausting all retries (empty if
        everything succeeded). Callers can use this to report or re-stage
        the corresponding files.
    """

    # Download a single file with retry + exponential backoff
    def _dl_one(url):
        attempt = 0
        clean_url = _sanitize_url(url)
        while True:
            try:
                # CASDA expects a list of URLs
                casda.download_files([clean_url], savedir=savedir)
                return True
            except Exception as e:
                attempt += 1
                print(f"Download failed [{attempt}] {url}: {_error_detail(e)}")

                if attempt >= max_retries:
                    return False

                # exponential backoff + jitter
                sleep_s = (backoff ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

    ok = 0
    failed_urls = []

    # Run downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_dl_one, u): u for u in urls}

        for fut in as_completed(futures):
            url = futures[fut]
            if fut.result():
                ok += 1
            else:
                failed_urls.append(url)

    print(f'>>> Downloaded {ok}/{len(urls)} files.'
          + (f' {len(failed_urls)} failed after retries.' if failed_urls else ''))
    return failed_urls