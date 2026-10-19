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

Colab Cell 1 prints live installation logs and the detected Python version. It
requires Python 3.10–3.12, just like the local app. If your runtime uses a newer
Python, select an available Python 3.12 runtime through **Runtime > Change runtime
type > Runtime version**. After installation completes, restart the runtime and
continue from Cell 2 without reinstalling packages. Restarting reloads libraries
that Colab may have imported before their versions were replaced.

For the pinned Gradio 4.42 UI, FastAPI is pinned to 0.112.2. FastAPI 0.113+
can fail request validation before a queued callback starts, even while the app
page loads correctly. To repair an already-installed Colab environment, run
`%pip install fastapi==0.112.2`, restart the session, and continue from Cell 2.

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
cloning the repository or importing `study_mitra.py`.
After editing the Python implementation or requirements, run `python sync_notebook.py`
to update the notebook's code cells and embedded installation requirements.

## Checks

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py study_mitra.py
```

The regression suite checks the real evaluation libraries, UI wiring, and an HTTP
file upload followed by queued transcription and session-specific summarization.
Model inference is stubbed to avoid large downloads. Actual transcription and model
quality require recordings, downloaded weights, and human references.
