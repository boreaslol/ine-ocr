"""Create a local owner-only credential file; never print the generated token."""

import os
from pathlib import Path
import secrets


def create_environment(target: Path):
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write("INE_OCR_API_BEARER_TOKEN=" + secrets.token_hex(32) + "\n")
    print("Created .env with a new local credential. Keep it private and out of Git.")


if __name__ == "__main__":
    create_environment(Path(__file__).resolve().parents[1] / ".env")
