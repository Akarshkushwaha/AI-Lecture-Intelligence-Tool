"""Local entry point for StudyMitra.ai."""

import argparse
from study_mitra import build_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StudyMitra.ai lecture intelligence")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio URL"
    )
    args = parser.parse_args()
    build_app().queue(max_size=5, default_concurrency_limit=1).launch(share=args.share)
