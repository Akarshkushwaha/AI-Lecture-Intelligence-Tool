# StudyMitra.ai — AI-Powered Lecture Intelligence 🎓

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akarshkushwaha/AI-Lecture-Intelligence-Tool/blob/main/lecture-intelligence-tool.ipynb)

StudyMitra turns recorded lectures into study material. Upload an audio/video file,
generate a timestamped transcript, then summarize it, identify concepts, ask questions,
or translate English content. Models run locally; no paid API key is required.

| Feature | Implementation |
| --- | --- |
| Transcription | faster-whisper / Whisper, language detection, VAD, word timestamps; automatic CPU/CUDA model selection |
| Summaries | facebook/bart-large-cnn with token-based map/reduce for long lectures; Brief, Medium, Detailed |
| Keywords and entities | KeyBERT, all-MiniLM-L6-v2, MMR diversity, spaCy English NER |
| Lecture Q&A | TF-IDF retrieval and deepset/roberta-base-squad2 extractive answers |
| Translation | Helsinki-NLP MarianMT, English to Hindi, French, German, Spanish, Arabic |
| Evaluation | Reference-based WER/MER/WIL, ROUGE, and SQuAD-style QA F1/Exact Match |

The UI uses Gradio. Python, PyTorch, Hugging Face Transformers, scikit-learn,
and FFmpeg provide the processing pipeline. MoviePy 2 is the audio extraction fallback.

## Run locally

Use Python **3.10–3.12** in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Install the FFmpeg executable and put it on PATH for normalized audio extraction.
On Ubuntu/Colab: `apt-get install ffmpeg`. `packages.txt` declares the system
package for compatible hosting environments. MoviePy's bundled FFmpeg can handle
supported inputs when the system executable is unavailable.

Open the local URL printed by Gradio. To explicitly create a public link:
`python app.py --share`. Colab's launch cell enables sharing.

## Run in Google Colab

Open the badge above, select a T4 GPU if available, and use **Run all**, or execute
Cells 1–11 in order. Colab can keep its default Python 3.13 runtime. No restart,
runtime downgrade, or skipped cells are needed.

Cell 1 creates a private Python 3.12 virtual environment, installing a compatible
interpreter automatically if necessary. It installs the complete Linux dependency
lock from `requirements-colab.lock` and checks imports including NumPy's compiled
extensions, pandas, spaCy, Gradio, and FastAPI in a separate process. Colab's own
NumPy/pandas packages are never replaced or imported by the application cells.
The server uses Matplotlib's non-interactive `Agg` backend and its own plotting
configuration directory, so Colab's `matplotlib_inline` setting cannot leak into
queued callbacks. The setup health check also renders a small figure.
Repeated setup reuses the same healthy environment; changed locks or failed
checks invalidate its ready marker.

Cells 2–10 keep the full source visible and write separate source files using
IPython's `%%writefile` magic. Rerunning a cell replaces its file. Cell 11 assembles
and executes those files in the verified environment, streams the server logs,
and prints the public Gradio URL. Keep Cell 11 running while using the app.
The first setup downloads Python packages, including GPU libraries, so it may take
several minutes; subsequent setup runs reuse them within the same Colab VM.

For the pinned Gradio 4.42 UI, FastAPI is pinned to 0.112.2. FastAPI 0.113+
can fail request validation before a queued callback starts, even while the app
page loads correctly. This version is included in the isolated dependency lock.

The first use of each feature downloads its model weights from Hugging Face;
subsequent runs use the cache. CPU processing is supported but slower. CUDA needs
a compatible PyTorch/CTranslate2 installation and NVIDIA libraries. Models remain
cached in memory, so restart the process to reclaim memory if using many models
on a small GPU. Automatic Whisper selection uses total GPU memory, not free memory.

## Workflow and limits

1. Upload MP4, MKV, WebM, MP3, WAV, FLAC, OGG, or M4A in **Transcribe**.
2. Other tabs use your session's transcript, or accept pasted text.
3. For **Evaluation**, supply human-verified references and predictions. Blank
   sections show N/A. Scores cannot be inferred from word count or confidence;
   different metrics are reported separately, without an invented overall accuracy.

Whisper supports multilingual transcription. BART summaries, English keywords/NER,
and the QA model are intended for English; translation accepts English source text.
Model output may contain errors and should be checked against the recording.
NER currently processes the first 100,000 characters; KeyBERT's document embedding
is limited by its encoder's context window. QA retrieves a subset of the transcript.

`app.py` launches the UI; `study_mitra.py` contains the shared pipeline and callbacks.
The Colab notebook contains the full implementation in separate, readable cells:
setup, configuration, audio extraction, transcription, summaries, keywords, Q&A,
translation, evaluation, Gradio compatibility, UI, and launch. It can run without
cloning the repository. Cell 11 assembles its visible source into `study_mitra.py`
inside the private runtime directory.
After editing the Python implementation or requirements, run `python sync_notebook.py`
to update the notebook's code cells and embedded runtime setup. After changing
`requirements.txt`, regenerate the Linux lock before the notebook:

```bash
uv pip compile requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28 --output-file requirements-colab.lock --no-annotate
python sync_notebook.py
```

## Checks

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py study_mitra.py
```

The regression suite checks the real evaluation libraries, UI wiring, and an HTTP
file upload followed by queued transcription and session-specific summarization.
Model inference is stubbed to avoid large downloads. Actual transcription and model
quality require recordings, downloaded weights, and human references.

The GitHub validation workflow additionally starts with Python 3.13 and NumPy 2
loaded, executes the notebook cells in order, repeats Cell 1, checks that the host
NumPy remains unchanged, and sends a real queued request to the isolated server.
It then runs the application regression suite in the private Python 3.12 runtime.
