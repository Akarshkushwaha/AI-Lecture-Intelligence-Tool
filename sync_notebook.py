"""Keep a full, standalone teaching notebook aligned with the local application.

Run `python sync_notebook.py` after changing study_mitra.py or requirements.txt.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "lecture-intelligence-tool.ipynb"
SECTION = re.compile(r"(?m)^# =+\n# CELL ([\d.]+): ([^\n]+)\n")
EXPLANATIONS = {
    "2": "Import libraries, select CPU/CUDA, and configure languages and model caching. No model weights are downloaded until a feature is used.",
    "3": "Normalize audio to 16 kHz mono PCM with FFmpeg, using MoviePy 2 as a fallback.",
    "4": "Load faster-whisper on demand and produce transcript text, timestamps, language information, and processing statistics.",
    "5": "Summarize token-sized lecture chunks with BART and repeatedly reduce their summaries without truncating the combined text.",
    "6": "Extract semantic keywords with KeyBERT/MMR and named entities with spaCy. NER processes the first 100,000 characters.",
    "7": "Retrieve relevant lecture context with TF-IDF and answer with RoBERTa across overlapping token windows. Includes reference-based QA metrics.",
    "8": "Translate English text into Hindi, French, German, Spanish, or Arabic using MarianMT and token-based chunking.",
    "9": "Calculate WER, ROUGE, and QA F1/Exact Match from human-verified references. No accuracy is inferred from transcript length.",
    "9.5": "Handle boolean JSON schemas used by the pinned Gradio client.",
    "10": "Define callbacks and build all six Gradio tabs. Each browser session has its own transcript state.",
}

# Embedded, not imported by Colab; only standard-library setup code runs in its kernel.
RUNTIME_ROOT = "/content/studymitra-runtime"
SETUP_HELPERS = (ROOT / "colab_runtime.py").read_text(encoding="utf-8")


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(runtime_root=RUNTIME_ROOT):
    source = (ROOT / "study_mitra.py").read_text(encoding="utf-8")
    sections = list(SECTION.finditer(source))
    if len(sections) != len(EXPLANATIONS):
        raise ValueError(
            "Application section headings changed; update notebook sections"
        )
    locked = (ROOT / "requirements-colab.lock").read_text(encoding="utf-8")
    setup = (
        "# CELL 1: ISOLATED DEPENDENCY SETUP (safe to rerun)\n"
        + SETUP_HELPERS
        + f"\nPROJECT_ROOT = Path({runtime_root!r})\n"
        + f"LOCKED_REQUIREMENTS = {locked!r}\n"
        + 'print(f"Notebook Python {sys.version.split()[0]}; application uses isolated Python 3.12.", flush=True)\n'
        + "APP_PYTHON = ensure_runtime(PROJECT_ROOT, LOCKED_REQUIREMENTS)\n"
        + 'if not shutil.which("ffmpeg"):\n'
        + '    if hasattr(os, "geteuid") and os.geteuid() == 0:\n'
        + '        run_setup("Update system package index", ["apt-get", "update", "-qq"])\n'
        + '        run_setup("Install FFmpeg", ["apt-get", "install", "-y", "ffmpeg"])\n'
        + "    else:\n"
        + '        print("System FFmpeg unavailable; audio extraction will use MoviePy’s bundled executable.", flush=True)\n'
        + 'print("Setup complete. Continue with Cell 2. No restart or skipped cells needed.", flush=True)\n'
    )
    cells = [
        cell(
            "markdown",
            "# StudyMitra.ai — AI Lecture Intelligence\n\n"
            "This notebook contains the **full implementation**, split into readable cells. "
            "Choose a T4 GPU if available, then use **Run all** or run Cells 1–11 in order. "
            "The notebook can use Colab’s default Python version.\n\n"
            "Cell 1 creates an isolated Python 3.12 environment. Cells 2–10 save the full "
            "source into that environment; Cell 11 executes it and starts the UI. "
            "No kernel restart, runtime downgrade, or skipped setup cell is required. "
            "First use downloads dependencies and model weights; evaluation requires reference data.\n",
        ),
        cell(
            "markdown",
            "## Cell 1 — Isolated dependency setup\n\n"
            "Install the locked app stack separately from Colab’s packages, then test its imports. "
            "Progress and errors appear below. It is safe to rerun this cell: a healthy matching "
            "environment is reused. Continue directly to Cell 2 when setup completes.\n",
        ),
        cell("code", setup),
    ]
    names = []
    for i, match in enumerate(sections):
        number, title = match.groups()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(source)
        name = f"{i + 2:02d}.py"
        names.append(name)
        cells.extend(
            [
                cell(
                    "markdown",
                    f"## Cell {number} — {title.title()}\n\n{EXPLANATIONS[number]}\n\n"
                    "This cell writes the full code below to the isolated app. It executes when Cell 11 launches.\n",
                ),
                cell(
                    "code",
                    f"%%writefile {runtime_root}/cells/{name}\n"
                    + source[match.start() : end],
                ),
            ]
        )
    cells.extend(
        [
            cell(
                "markdown",
                "## Cell 11 — Launch StudyMitra\n\n"
                "Execute the code from Cells 2–10 in the verified app environment. "
                "Open the public Gradio URL printed below. This cell stays running to serve the app "
                "and show its logs; stop it with the square button when finished.\n",
            ),
            cell(
                "code",
                "# CELL 11: LAUNCH THE ISOLATED APPLICATION\n"
                + f"launch_notebook(PROJECT_ROOT, {names!r}, share=True)\n",
            ),
        ]
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {
                "name": "lecture-intelligence-tool.ipynb",
                "provenance": [],
                "gpuType": "T4",
                "toc_visible": True,
            },
            "accelerator": "GPU",
        },
        "cells": [dict(c, id=f"study-mitra-{i:02d}") for i, c in enumerate(cells)],
    }


if __name__ == "__main__":
    notebook = build_notebook()
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(notebook['cells'])} cells including the full implementation.")
