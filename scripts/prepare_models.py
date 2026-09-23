"""Download public base models at build time, then verify their fixed hashes."""

from ine_ocr.document_ocr import PaddleLineReader
from verify_runtime import verify_assets


def main():
    PaddleLineReader("small").ready()
    verified = verify_assets()
    print({"public_assets_verified": len(verified)})


if __name__ == "__main__":
    main()
