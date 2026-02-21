import os
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    """

    # Download a single file with retry + exponential backoff
    def _dl_one(url):
        attempt = 0
        while True:
            try:
                # CASDA expects a list of URLs
                casda.download_files([url], savedir=savedir)
                return True
            except Exception as e:
                attempt += 1
                print(f"Download failed [{attempt}] {url}: {e}")

                if attempt >= max_retries:
                    return False

                # exponential backoff + jitter
                sleep_s = (backoff ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

    ok = 0

    # Run downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_dl_one, u): u for u in urls}

        for fut in as_completed(futures):
            if fut.result():
                ok += 1

    print(f'>>> Downloaded {len(urls)} files.')
    return ok