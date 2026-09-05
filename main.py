#!/usr/bin/env python3
"""Thin entry point so the project can be run without installing it:

    python main.py image --input photo.jpg --output out.jpg
    python main.py batch --input-dir photos/ --output-dir results/
    python main.py webcam --camera 0

Once installed (`pip install .`), the same functionality is available
as the `facemesh` console command.
"""
from src.facemesh.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
