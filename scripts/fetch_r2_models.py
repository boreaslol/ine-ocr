"""Download one immutable R2 release asset and install only its verified allowlist."""

import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ine_ocr.r2_profile import contract, model_root, verify_models


def unpack_verified(archive, destination, expected):
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("r2_archive_members_mismatch")
        for member in members:
            name = PurePosixPath(member.name)
            entry = expected[member.name]
            if name.is_absolute() or ".." in name.parts or not member.isfile() or member.size != entry["bytes"]:
                raise RuntimeError("r2_archive_member_invalid")
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as incoming, target.open("xb") as output:
                while chunk := incoming.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise RuntimeError("r2_archive_file_hash_mismatch")


def download(target, entry):
    for attempt in range(3):
        try:
            started = time.monotonic()
            digest = hashlib.sha256()
            size = 0
            with urlopen(entry["url"], timeout=30) as response, target.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > entry["bytes"] or time.monotonic() - started > 300:
                        raise RuntimeError("r2_download_limit_exceeded")
                    output.write(chunk)
                    digest.update(chunk)
            if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise RuntimeError("r2_download_hash_mismatch")
            return
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError("r2_download_unavailable") from None
            time.sleep(5 * (attempt + 1))


def main():
    destination = model_root()
    if destination.exists():
        verify_models(destination)
        print("R2 model assets already verified")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = contract()
    with tempfile.TemporaryDirectory(prefix="r2-install-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "models.tar.gz"
        staged = temporary / "models"
        staged.mkdir()
        download(archive, profile["archive"])
        unpack_verified(archive, staged, profile["files"])
        verify_models(staged)
        os.replace(staged, destination)
    print("R2 model assets downloaded and verified")


if __name__ == "__main__":
    main()
