"""HTTP plumbing for any source that fetches documents over the network.

A shared session retries transient failures, and `download_to()` streams a file to disk
atomically while hashing it. Keep network calls in source-specific modules built on this,
so tests can monkeypatch `requests.Session.get` in one place.
"""

import hashlib
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT = 20
DOWNLOAD_TIMEOUT = 60

# Transient failures (rate limiting, 5xx) are retried with backoff: 1s, 2s, 4s, ...
RETRY = Retry(
    total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET"]
)


def make_session():
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=RETRY))
    return session


SESSION = make_session()


def get_json(url, params=None, headers=None):
    response = SESSION.get(url, params=params, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# Stream a file to `target` and return its sha256. It is written to <target>.part and
# renamed at the end, so an interrupted download never looks like a finished file.
def download_to(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with SESSION.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(partial, "wb") as out:
            for block in response.iter_content(chunk_size=1 << 16):
                digest.update(block)
                out.write(block)
    os.replace(partial, target)
    return digest.hexdigest()
