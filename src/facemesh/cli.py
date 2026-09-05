"""CLI implementation, importable both as `python main.py` and as an
installed console script (`facemesh`) after `pip install .`.
"""

from __future__ import annotations

import argparse
import sys

from .exceptions import FaceMeshError
from .image_processor import process_directory, process_image
from .logging_config import get_logger
from .video_processor import run_webcam

logger = get_logger("facemesh.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facemesh",
        description="Face Mesh Detection with MediaPipe (468/478 landmarks).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_image = subparsers.add_parser("image", help="Process a single static image")
    p_image.add_argument("--input", required=True, help="Path to input image")
    p_image.add_argument("--output", help="Path to save annotated image")
    p_image.add_argument("--landmarks-json", help="Path to save raw landmark JSON")
    p_image.add_argument("--max-faces", type=int, default=5)

    p_batch = subparsers.add_parser("batch", help="Process a directory of images")
    p_batch.add_argument("--input-dir", required=True)
    p_batch.add_argument("--output-dir", required=True)
    p_batch.add_argument("--max-faces", type=int, default=5)
    p_batch.add_argument("--no-landmarks-json", action="store_true")

    p_cam = subparsers.add_parser("webcam", help="Run live detection on a webcam")
    p_cam.add_argument("--camera", type=int, default=0)
    p_cam.add_argument("--max-faces", type=int, default=1)
    p_cam.add_argument("--no-mirror", action="store_true")
    p_cam.add_argument("--record", help="Path to save annotated video (e.g. out.mp4)")
    p_cam.add_argument("--no-fps", action="store_true")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "image":
            result = process_image(
                input_path=args.input,
                output_path=args.output,
                landmarks_json_path=args.landmarks_json,
                max_num_faces=args.max_faces,
            )
            print(f"Detected {result.face_count} face(s).")

        elif args.command == "batch":
            written = process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                max_num_faces=args.max_faces,
                save_landmarks_json=not args.no_landmarks_json,
            )
            print(f"Processed {len(written)} image(s) into {args.output_dir}")

        elif args.command == "webcam":
            run_webcam(
                camera_index=args.camera,
                max_num_faces=args.max_faces,
                mirror=not args.no_mirror,
                output_video_path=args.record,
                show_fps=not args.no_fps,
            )

        return 0

    except FaceMeshError as exc:
        logger.error("%s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
