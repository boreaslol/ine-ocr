"""Start only the complete frozen R2 profile; never fall back to base weights."""

from ine_ocr.r2_profile import serve as main


if __name__ == "__main__":
    main()
