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


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    source = (ROOT / "study_mitra.py").read_text(encoding="utf-8")
    sections = list(SECTION.finditer(source))
    if len(sections) != len(EXPLANATIONS):
        raise ValueError("Application section headings changed; update notebook sections")
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    setup = (
        "# CELL 1: DEPENDENCY INSTALLATION (Google Colab)\n"
        "import subprocess\nimport sys\n\n"
        "# Matches requirements.txt; no repository clone or external Python source is needed.\n"
        f"requirements = {json.dumps(requirements, indent=4)}\n"
        'subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)\n'
        'subprocess.run(["apt-get", "update", "-qq"], check=True)\n'
        'subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)\n'
        'print("Setup complete. If libraries were already imported, restart the runtime before continuing from Cell 2.")\n'
    )
    cells = [
        cell("markdown", "# StudyMitra.ai — AI Lecture Intelligence\n\n"
             "This notebook contains the **full implementation**, split into readable cells. "
             "Run the cells in order: setup, configuration, audio extraction, transcription, "
             "summarization, keywords, Q&A, translation, evaluation, UI, then launch.\n\n"
             "In Colab, select a GPU runtime for faster processing. First use of each model "
             "downloads its weights. Evaluation requires human-verified references.\n"),
        cell("markdown", "## Cell 1 — Dependency installation\n\n"
             "Install the compatible Python stack, spaCy English model, and system FFmpeg. "
             "Skip this cell after a runtime restart if installation has already completed.\n"),
        cell("code", setup),
    ]
    for i, match in enumerate(sections):
        number, title = match.groups()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(source)
        cells.extend([
            cell("markdown", f"## Cell {number} — {title.title()}\n\n{EXPLANATIONS[number]}\n"),
            cell("code", source[match.start():end]),
        ])
    cells.extend([
        cell("markdown", "## Cell 11 — Launch StudyMitra\n\n"
             "Build the UI from the functions above and create a public Gradio URL. "
             "The notebook must remain running while the URL is in use.\n"),
        cell("code", '# CELL 11: LAUNCH GRADIO\n'
             'gr.close_all()\n'
             'demo = build_app()\n'
             'demo.queue(max_size=5, default_concurrency_limit=1).launch(share=True)\n'),
    ])
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "lecture-intelligence-tool.ipynb", "provenance": [],
                      "gpuType": "T4", "toc_visible": True},
            "accelerator": "GPU",
        },
        "cells": [dict(c, id=f"study-mitra-{i:02d}") for i, c in enumerate(cells)],
    }


if __name__ == "__main__":
    notebook = build_notebook()
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(notebook['cells'])} cells including the full implementation.")
