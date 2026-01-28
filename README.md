# AI-Powered Lecture Intelligence Tool 🎓
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akarshkushwaha/AI-Lecture-Intelligence-Tool/blob/main/Copy_of_Updated_Lecture_Intelligence_Tool.ipynb)

## Overview
This tool converts video lectures into concise, study-ready notes. It utilizes advanced NLP models to transcribe audio, summarize content, and extract key takeaways, helping students save time and improve retention.

## Features
* **Video Transcription:** Uses OpenAI's `Whisper` (via Faster-Whisper) for high-accuracy speech-to-text.
* **Intelligent Summarization:** Uses Hugging Face Transformers to condense transcripts into abstractive summaries.
* **Keyword Extraction:** Identifies core concepts automatically.
* **Interactive UI:** Built with Gradio for a seamless web-based experience.

## Tech Stack
* **Python**
* **Hugging Face Transformers**
* **Gradio**
* **Torch / PyTorch**
* **FFmpeg**

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python app.py`
