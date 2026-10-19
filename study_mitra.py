# ============================================================
# CELL 2: IMPORTS & GLOBAL CONFIGURATION
# ============================================================

import os, gc, re, json, time, warnings, subprocess, logging, sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

logging.basicConfig(level=logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import pandas as pd

import nltk
import spacy
from nltk.tokenize import PunktSentenceTokenizer

sent_tokenize = PunktSentenceTokenizer().tokenize

import jiwer
from rouge_score import rouge_scorer as rouge_scorer_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Device Configuration ──────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print("=" * 55)
print("  STUDY MITRA - AI Lecture Intelligence Platform")
print("=" * 55)
print(f"\nDevice     : {DEVICE.upper()}")
print(f"Compute    : {COMPUTE_TYPE}")

if DEVICE == "cuda":
    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_memory / 1e9
    print(f"GPU        : {gpu.name}")
    print(f"VRAM       : {vram_gb:.1f} GB")
    if vram_gb < 10:
        print("WARNING: Low VRAM - will use medium model for Whisper")
else:
    print("CPU mode selected (processing may be slower)")

# ── Lazy Model Registry ───────────────────────────────────────
# All models loaded on first use. Never loaded at import time.
MODELS: Dict[str, Any] = {}

# ── Language Configuration ────────────────────────────────────
LANG_MAP = {
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Arabic": "ar",
}

TRANSLATION_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
}

# ── spaCy NER ─────────────────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_sm")
    print(f"\nspaCy      : {spacy.__version__} (en_core_web_sm loaded)")
except OSError:
    print(
        "WARNING: spaCy model not found - run: python -m spacy download en_core_web_sm"
    )
    NLP = None

print(f"PyTorch    : {torch.__version__}")
print("\n[OK] Configuration complete - all modules imported")
print("     Models will load lazily when each feature is first called")


# ============================================================
# CELL 3: AUDIO EXTRACTION - FFmpeg primary, MoviePy fallback
# ============================================================

from pathlib import Path
import subprocess

SUPPORTED_FORMATS = {".mp4", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def check_ffmpeg() -> bool:
    """Check if FFmpeg binary is available on the system PATH."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_audio_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Extract & normalize audio using FFmpeg.

    Command flags used:
        -y              : Overwrite output without prompting
        -i input_path   : Input file
        -vn             : Strip video stream (not needed for ASR)
        -ar 16000       : Resample to 16 kHz (Whisper optimal rate)
        -ac 1           : Convert to mono channel
        -af loudnorm    : EBU R128 loudness normalization (I=-23, LRA=7, TP=-2)
        -acodec pcm_s16le : 16-bit signed little-endian PCM WAV output
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-af",
        "loudnorm=I=-23:LRA=7:TP=-2",
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


def extract_audio_moviepy(input_path: str, output_path: str) -> bool:
    """
    Fallback audio extraction using MoviePy.
    Used only when FFmpeg is unavailable. No loudnorm applied.
    """
    clip = audio = None
    try:
        from moviepy import VideoFileClip, AudioFileClip

        ext = Path(input_path).suffix.lower()
        clip = VideoFileClip(input_path) if ext in {".mp4", ".mkv", ".webm"} else None
        audio = clip.audio if clip else AudioFileClip(input_path)
        if audio is None:
            return False
        audio.write_audiofile(
            output_path, fps=16000, nbytes=2, ffmpeg_params=["-ac", "1"]
        )
        return True
    except Exception as e:
        print(f"[ERROR] MoviePy fallback failed: {e}")
        return False
    finally:
        if audio is not None:
            audio.close()
        if clip is not None:
            clip.close()


def extract_audio(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Main audio extraction entry point.

    Tries FFmpeg first (fast, full normalization).
    Falls back to MoviePy if FFmpeg is not available.

    Args:
        input_path:  Path to input audio or video file
        output_path: Destination .wav file path

    Returns:
        str: Path to the extracted WAV file

    Raises:
        ValueError:   Unsupported file format
        FileNotFoundError: Input file missing
        RuntimeError: Both extraction methods failed
    """
    if output_path is None:
        import tempfile

        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_FORMATS}")
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    print(f"[Audio] Input : {Path(input_path).name}")

    if check_ffmpeg():
        print("[Audio] Method: FFmpeg + EBU R128 normalization")
        if extract_audio_ffmpeg(input_path, output_path):
            mb = Path(output_path).stat().st_size / 1_000_000
            print(f"[Audio] Done  : {mb:.2f} MB | 16kHz mono PCM")
            return output_path
        print("[Audio] FFmpeg failed, falling back to MoviePy...")
    else:
        print("[Audio] FFmpeg not found, using MoviePy (slower, no normalization)")

    if extract_audio_moviepy(input_path, output_path):
        print(f"[Audio] Done (MoviePy): {output_path}")
        return output_path

    raise RuntimeError("Audio extraction failed with both FFmpeg and MoviePy.")


print("[OK] Audio extraction module ready")
print(f"     FFmpeg available : {check_ffmpeg()}")
print(f"     Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")


# ============================================================
# CELL 4: SPEECH-TO-TEXT - faster-whisper large-v3
# ============================================================

from faster_whisper import WhisperModel
from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    """Structured container for one transcript segment with timing."""

    start: float
    end: float
    text: str
    words: list = field(default_factory=list)

    def duration(self) -> float:
        return self.end - self.start

    def timestamp_str(self) -> str:
        def fmt(t):
            return f"{int(t // 60):02d}:{t % 60:05.2f}"

        return f"[{fmt(self.start)} --> {fmt(self.end)}]"


def get_whisper_model_size() -> str:
    """
    Auto-select Whisper model based on available VRAM.

    VRAM thresholds:
        large-v3 float16 --> needs ~4 GB VRAM (T4 has 15 GB - safe)
        medium   float16 --> needs ~2.5 GB VRAM
        base     int8    --> CPU only, ~150 MB RAM
    """
    if DEVICE != "cuda":
        return "base"
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 8:
            return "large-v3"
        elif vram >= 5:
            return "medium"
        else:
            return "small"
    except Exception:
        return "base"


def load_whisper_model(model_size: Optional[str] = None) -> WhisperModel:
    """
    Lazily load and cache faster-whisper model.

    CTranslate2 key optimizations:
        float16 - Half-precision inference to reduce VRAM usage
        int8    - 8-bit quantized: 4x VRAM savings, slight accuracy trade-off on CPU
    """
    if model_size is None:
        model_size = get_whisper_model_size()

    key = f"whisper_{model_size}"
    if key not in MODELS:
        sizes_gb = {"large-v3": 3.0, "medium": 1.5, "small": 0.5, "base": 0.15}
        gb = sizes_gb.get(model_size, 1.0)
        print(f"[Whisper] Loading '{model_size}' on {DEVICE.upper()} ({COMPUTE_TYPE})")
        print(f"[Whisper] Download size ~{gb:.1f} GB (cached after first run)")
        MODELS[key] = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=4,
            num_workers=2,
        )
        print(f"[Whisper] Loaded and cached as MODELS['{key}']")
    return MODELS[key]


def transcribe_audio(
    audio_path: str,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    task: str = "transcribe",
) -> Dict[str, Any]:
    """
    Full speech-to-text pipeline using faster-whisper.

    Steps:
        1. Load/retrieve model from MODELS cache (lazy)
        2. Apply Silero VAD filter to remove silence
        3. Transcribe with beam search (beam_size=5)
        4. Collect segment and word timestamps
        5. Release unused allocator memory (models remain cached)

    Args:
        audio_path      : Path to 16kHz mono WAV file
        model_size      : None = auto-select based on VRAM
        language        : ISO code to force language (None = auto-detect)
        beam_size       : Beam search width
        vad_filter      : Enable Silero VAD silence removal
        word_timestamps : Include per-word start/end/probability
        task            : 'transcribe' or 'translate' (to English)

    Returns dict:
        transcript    - Full text string
        segments      - List[TranscriptSegment]
        timestamped   - Human-readable timestamped transcript
        language      - Detected ISO language code
        lang_prob     - Language detection confidence (0-1)
        duration      - Audio length in seconds
        num_words     - Total word count
        rtf           - Real-Time Factor (elapsed / audio_duration)
    """
    model = load_whisper_model(model_size)
    print(f"\n[Transcribe] File   : {Path(audio_path).name}")
    print(
        f"[Transcribe] Config : beam={beam_size}, VAD={vad_filter}, lang={language or 'auto'}"
    )

    t0 = time.time()
    segments_gen, info = model.transcribe(
        audio_path,
        task=task,
        language=language,
        beam_size=beam_size,
        # Voice Activity Detection - removes silence
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
            "threshold": 0.5,
        },
        # Word-level timestamps
        word_timestamps=word_timestamps,
        # Hallucination prevention
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        # Context conditioning for better continuity
        condition_on_previous_text=True,
    )

    segments, text_parts = [], []
    for seg in segments_gen:
        words = []
        if word_timestamps and seg.words:
            words = [
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "prob": round(w.probability, 3),
                }
                for w in seg.words
            ]
        s = TranscriptSegment(
            start=round(seg.start, 2),
            end=round(seg.end, 2),
            text=seg.text.strip(),
            words=words,
        )
        segments.append(s)
        text_parts.append(seg.text.strip())

    elapsed = time.time() - t0
    full_text = " ".join(text_parts)
    timestamped = "\n".join(f"{s.timestamp_str()} {s.text}" for s in segments)

    # Free VRAM after heavy inference
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    rtf = elapsed / max(info.duration, 1)
    print(f"\n[Transcribe] Done     : {elapsed:.1f}s elapsed")
    print(f"[Transcribe] Language : {info.language} ({info.language_probability:.1%})")
    print(
        f"[Transcribe] Duration : {info.duration:.1f}s | {len(segments)} segs | {len(full_text.split())} words"
    )
    print(f"[Transcribe] RTF      : {rtf:.3f}  (lower = faster than realtime)")

    return {
        "transcript": full_text,
        "segments": segments,
        "timestamped": timestamped,
        "language": info.language,
        "lang_prob": round(info.language_probability, 4),
        "duration": round(info.duration, 2),
        "num_segments": len(segments),
        "num_words": len(full_text.split()),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 3),
    }


print("[OK] Transcription module ready")
print(f"     Auto-selected model: {get_whisper_model_size()}")


# ============================================================
# CELL 5: SUMMARIZATION - BART-large-CNN + Map-Reduce
# ============================================================

from transformers import BartForConditionalGeneration, BartTokenizer

BART_MODEL = "facebook/bart-large-cnn"
MAX_CHUNK_TOKENS = 900  # BART context limit is 1024; leave room for generation
OVERLAP_TOKENS = 150  # Overlap prevents losing context at chunk boundaries


def load_bart():
    """Lazily load BART model and tokenizer into MODELS cache."""
    if "bart" not in MODELS:
        print(f"[BART] Loading {BART_MODEL}...")
        print("[BART] Download ~1.6 GB on first run (cached after)")
        tok = BartTokenizer.from_pretrained(BART_MODEL)
        mdl = BartForConditionalGeneration.from_pretrained(BART_MODEL)
        mdl = mdl.to(DEVICE)
        MODELS["bart"] = (mdl, tok)
        print(f"[BART] Loaded on {DEVICE.upper()} - cached in MODELS['bart']")
    return MODELS["bart"]


def tokenizer_chunk(text: str, tokenizer, max_tok: int, overlap: int) -> List[str]:
    """
    Split text into overlapping token-based chunks using BartTokenizer.

    Algorithm:
        1. Encode full text -> subword token IDs
        2. Slide window of max_tok tokens with (max_tok - overlap) step
        3. Decode each window back to readable text

    Why tokenizer-based over character/word splitting?
        - BART's 1024-token limit is in subword tokens, not words
        - 'unbelievable' = 4 tokens; word-splitting would under-count
        - Tokenizer decoding guarantees no partial tokens
    """
    if max_tok <= 0 or not 0 <= overlap < max_tok:
        raise ValueError("Require max_tok > 0 and 0 <= overlap < max_tok")
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, start, step = [], 0, max_tok - overlap
    while start < len(ids):
        end = min(start + max_tok, len(ids))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids):
            break
        start += step
    return chunks


def summarize_one_chunk(
    chunk: str, model, tokenizer, max_len: int = 200, min_len: int = 50
) -> str:
    """Run BART inference on a single text chunk."""
    inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(
        DEVICE
    )
    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            length_penalty=2.0,  # Encourages completeness
            early_stopping=True,
            no_repeat_ngram_size=3,  # Prevents repetitive phrases
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize_text(text: str, detail: str = "medium") -> Dict[str, Any]:
    """
    Abstractive summarization using BART with Map-Reduce strategy.

    MAP  : Split full transcript into N overlapping chunks -> summarize each
    REDUCE: If combined chunk summaries too long -> summarize again recursively

    This ensures the ENTIRE lecture is summarized, not just the first 1024 tokens.

    Args:
        text   : Full transcript text
        detail : 'brief' | 'medium' | 'detailed'

    Returns dict:
        summary           - Final summary string
        chunk_summaries   - Individual chunk summaries (for debugging)
        num_chunks        - Number of chunks processed
        compression_ratio - input_words / output_words
    """
    cfg = {"brief": (130, 40), "medium": (200, 60), "detailed": (300, 80)}
    max_len, min_len = cfg.get(detail, (200, 60))

    if not text.strip():
        raise ValueError("Provide text to summarize")
    model, tokenizer = load_bart()

    print(f"\n[BART] Input: {len(text.split())} words | detail={detail}")

    # ── MAP PHASE ─────────────────────────────────────────────
    chunks = tokenizer_chunk(text, tokenizer, MAX_CHUNK_TOKENS, OVERLAP_TOKENS)
    print(
        f"[BART] Chunks: {len(chunks)} (each ~{MAX_CHUNK_TOKENS} tokens, {OVERLAP_TOKENS}-tok overlap)"
    )

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[BART] Summarizing chunk {i+1}/{len(chunks)}...", end="\r")
        chunk_summaries.append(
            summarize_one_chunk(chunk, model, tokenizer, max_len, min_len)
        )

    combined = " ".join(chunk_summaries)

    # ── REDUCE PHASE ──────────────────────────────────────────
    combined_ids = tokenizer.encode(combined, add_special_tokens=False)
    while len(combined_ids) > MAX_CHUNK_TOKENS:
        reduced = [
            summarize_one_chunk(c, model, tokenizer, max_len, min_len)
            for c in tokenizer_chunk(combined, tokenizer, MAX_CHUNK_TOKENS, 0)
        ]
        combined = " ".join(reduced)
        new_ids = tokenizer.encode(combined, add_special_tokens=False)
        if len(new_ids) >= len(combined_ids):
            raise RuntimeError("Summary reduction did not shrink; try Brief detail.")
        combined_ids = new_ids
    final = combined

    # Free VRAM
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    ratio = len(text.split()) / max(len(final.split()), 1)
    print(
        f"\n[BART] Done: {len(text.split())} -> {len(final.split())} words ({ratio:.1f}x compression)"
    )

    return {
        "summary": final,
        "chunk_summaries": chunk_summaries,
        "num_chunks": len(chunks),
        "input_words": len(text.split()),
        "output_words": len(final.split()),
        "compression_ratio": round(ratio, 1),
    }


print("[OK] Summarization module ready")
print(f"     Model: {BART_MODEL}")
print("     Strategy: Map-Reduce with 900-token chunks, 150-token overlap")


# ============================================================
# CELL 6: KEYWORD EXTRACTION - KeyBERT + MMR + spaCy NER
# ============================================================

NER_LABELS = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "PRODUCT", "LAW", "WORK_OF_ART"}


def load_keybert() -> Any:
    """Lazily load KeyBERT with all-MiniLM-L6-v2 sentence embeddings."""
    if "keybert" not in MODELS:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        print("[KeyBERT] Loading all-MiniLM-L6-v2 sentence embeddings...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        MODELS["keybert"] = KeyBERT(model=st_model)
        print("[KeyBERT] Loaded and cached in MODELS['keybert']")
    return MODELS["keybert"]


def extract_keywords_entities(
    text: str,
    top_n: int = 15,
    ngram_range: tuple = (1, 3),
    diversity: float = 0.7,
) -> Dict[str, Any]:
    """
    Extract semantic keywords and named entities from lecture transcript.

    KeyBERT + MMR Pipeline:
        1. Embed document with all-MiniLM-L6-v2 (384-dim vectors)
        2. Generate candidate n-grams (1-word to 3-word phrases)
        3. Embed each candidate
        4. Score by cosine similarity(candidate, document)
        5. Re-rank with MMR:
              score = diversity*sim(ki,doc) - (1-diversity)*max_j sim(ki,kj)
           diversity=0.7 balances relevance (0.3 weight) vs uniqueness (0.7 weight)

    spaCy NER:
        - Uses en_core_web_sm dependency parse + NER pipeline
        - Processes first 100K chars (spaCy default limit)
        - Groups entities by type: PERSON, ORG, GPE, DATE, etc.

    Args:
        text       : Full transcript text
        top_n      : Number of keywords to return
        ngram_range: (min_n, max_n) for candidate phrase length
        diversity  : MMR diversity 0=no variety, 1=max variety

    Returns dict:
        keywords : List of {keyword, score} sorted by relevance
        entities : Dict of {entity_type: [entity_list]}
        topic    : Top keyword (proxy for main topic)
    """
    kw_model = load_keybert()

    print(f"[KeyBERT] Extracting top {top_n} keywords (diversity={diversity})...")

    # ── KeyBERT + MMR ────────────────────────────────────────
    kws = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words="english",
        use_mmr=True,
        diversity=diversity,
        top_n=top_n,
    )

    # ── spaCy NER ─────────────────────────────────────────────
    entities: Dict[str, List[str]] = {}
    if NLP is not None:
        doc = NLP(text[:100_000])  # spaCy processes up to 100K chars
        for ent in doc.ents:
            if ent.label_ in NER_LABELS:
                label = ent.label_
                if label not in entities:
                    entities[label] = []
                if ent.text not in entities[label]:
                    entities[label].append(ent.text)
    else:
        print("[KeyBERT] WARNING: spaCy not loaded - skipping NER")

    n_entities = sum(len(v) for v in entities.values())
    print(f"[KeyBERT] Done: {len(kws)} keywords, {n_entities} named entities")

    return {
        "keywords": [{"keyword": kw, "score": round(sc, 4)} for kw, sc in kws],
        "entities": entities,
        "topic": kws[0][0] if kws else "N/A",
    }


def format_keywords(result: Dict) -> str:
    """Format keyword results as a readable table."""
    lines = [
        "KEYWORD EXTRACTION RESULTS",
        "-" * 50,
        f"{'Rank':<5} {'Keyword':<30} {'Score':<8} Bar",
    ]
    lines.append("-" * 50)
    for i, kw in enumerate(result["keywords"], 1):
        bar = "#" * int(kw["score"] * 25)
        lines.append(f"{i:<5} {kw['keyword']:<30} {kw['score']:.4f}  {bar}")
    lines.append("")
    lines.append("NAMED ENTITIES:")
    for label, ents in result["entities"].items():
        lines.append(f"  {label:<15}: {', '.join(ents[:8])}")
    return "\n".join(lines)


print("[OK] Keyword extraction module ready")
print("     Primary: KeyBERT + MMR (all-MiniLM-L6-v2)")
print("     Secondary: spaCy en_core_web_sm NER")


# ============================================================
# CELL 7: QUESTION ANSWERING - RoBERTa-SQuAD2 + TF-IDF Retrieval
# ============================================================

from transformers import pipeline as hf_pipeline

QA_MODEL = "deepset/roberta-base-squad2"


def load_qa_pipeline():
    """Lazily load extractive QA pipeline into MODELS cache."""
    if "qa" not in MODELS:
        print(f"[QA] Loading {QA_MODEL}...")
        print("[QA] Download ~500 MB on first run (cached after)")
        MODELS["qa"] = hf_pipeline(
            "question-answering",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            handle_impossible_answer=True,  # SQuAD2 unanswerable support
        )
        print(f"[QA] Loaded on {DEVICE.upper()} - cached in MODELS['qa']")
    return MODELS["qa"]


def retrieve_context_tfidf(
    transcript: str,
    question: str,
    top_k: int = 3,
    chunk_words: int = 200,
) -> Tuple[str, List[str]]:
    """
    Retrieve the most relevant transcript chunks for a question using TF-IDF.

    Algorithm:
        1. Split transcript into overlapping sentence-based chunks (~200 words)
        2. TF-IDF vectorize all chunks + the question (bigrams, 5000 features)
        3. Compute cosine_similarity(question_vector, chunk_vectors)
        4. Select top-k chunks by similarity score
        5. Return chunks in their original order (preserves context flow)

    Args:
        transcript  : Full lecture transcript
        question    : User's question string
        top_k       : Number of chunks to retrieve
        chunk_words : Approximate words per chunk

    Returns:
        (context_string, list_of_retrieved_chunks)
    """
    sents = sent_tokenize(transcript)
    if not sents:
        return transcript[:2000], [transcript[:2000]]

    # Build overlapping chunks from sentences
    chunks, buf, buf_len = [], [], 0
    for sent in sents:
        w = len(sent.split())
        if buf_len + w > chunk_words and buf:
            chunks.append(" ".join(buf))
            buf = buf[-2:]  # 2-sentence overlap
            buf_len = sum(len(s.split()) for s in buf)
        buf.append(sent)
        buf_len += w
    if buf:
        chunks.append(" ".join(buf))

    if len(chunks) == 1:
        return chunks[0], chunks

    # TF-IDF similarity
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", max_features=5000
    )
    try:
        tfidf = vectorizer.fit_transform(chunks + [question])
    except ValueError:
        selected = chunks[: max(1, top_k)]
        return " ".join(selected), selected
    sims = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]

    # Top-k in original order
    top_idx = sorted(np.argsort(sims)[::-1][:top_k].tolist())
    top_chunks = [chunks[i] for i in top_idx]
    return " ".join(top_chunks), top_chunks


def answer_question(
    transcript: str,
    question: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Answer a question based on the lecture transcript.

    Full Pipeline:
        1. TF-IDF retrieval: find top-3 relevant transcript chunks
        2. Use overlapping token windows across all retrieved context
        3. Run RoBERTa-SQuAD2 extractive QA on those windows
        4. Return answer + confidence + retrieved context

    Why Extractive QA (not generative)?
        - Extractive QA returns spans from the actual transcript (verifiable)
        - Generative QA can hallucinate facts not in the lecture
        - RoBERTa-SQuAD2 F1=82.9% on unanswerable questions dataset

    Args:
        transcript     : Full lecture transcript
        question       : User question string
        top_k          : Number of TF-IDF chunks to use as context

    Returns dict:
        answer          - Extracted answer span
        score           - Model confidence (0-1)
        confidence_pct  - Formatted percentage
        context         - Context used for QA
        chunks_used     - Individual retrieved chunks
    """
    if not transcript.strip():
        return {
            "answer": "No transcript loaded.",
            "score": 0.0,
            "confidence_pct": "0.0%",
            "context": "",
            "chunks_used": [],
        }
    if not question.strip():
        return {
            "answer": "Please type a question.",
            "score": 0.0,
            "confidence_pct": "0.0%",
            "context": "",
            "chunks_used": [],
        }

    qa = load_qa_pipeline()

    print(f"[QA] Question: {question[:80]}...")
    context, chunks = retrieve_context_tfidf(transcript, question, top_k)

    # The QA pipeline uses overlapping token windows over all retrieved context.

    print(
        f"[QA] Context : {len(context.split())} words from {len(chunks)} retrieved chunks"
    )

    result = qa(
        question=question,
        context=context,
        max_answer_len=120,
        max_seq_len=512,
        doc_stride=128,
        handle_impossible_answer=True,
    )
    score = result["score"]
    answer = result["answer"]

    # Low confidence = unanswerable
    if not answer.strip() or score < 0.01:
        answer = "The answer to this question was not found in the transcript."

    print(f"[QA] Answer  : {answer[:100]} (confidence: {score:.1%})")

    return {
        "answer": answer,
        "score": round(score, 4),
        "confidence_pct": f"{score:.1%}",
        "context": context,
        "chunks_used": chunks,
    }


def compute_qa_metrics(
    predictions: List[str], ground_truths: List[str]
) -> Dict[str, float]:
    """
    Compute token-level F1 and Exact Match for QA evaluation.

    Token F1 = 2 * precision * recall / (precision + recall)
    where precision/recall are over shared TOKEN bags (lowercased, no punctuation).

    This matches the official SQuAD evaluation script methodology.
    """
    from collections import Counter
    import string

    if not predictions or len(predictions) != len(ground_truths):
        raise ValueError("Provide equal, non-empty lists of predictions and references")

    def normalize(text):
        text = "".join(c for c in text.lower() if c not in string.punctuation)
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

    em_scores, f1_scores = [], []
    for pred, truth in zip(predictions, ground_truths):
        p, t = normalize(pred), normalize(truth)
        p_toks, t_toks = p.split(), t.split()
        em_scores.append(float(p == t))
        common = sum((Counter(p_toks) & Counter(t_toks)).values())
        if not p_toks or not t_toks:
            f1_scores.append(float(p_toks == t_toks))
        elif not common:
            f1_scores.append(0.0)
        else:
            prec, rec = common / len(p_toks), common / len(t_toks)
            f1_scores.append(2 * prec * rec / (prec + rec))
    return {
        "exact_match": round(sum(em_scores) / len(em_scores) * 100, 2),
        "f1": round(sum(f1_scores) / len(f1_scores) * 100, 2),
        "n_samples": len(predictions),
    }


print("[OK] Question Answering module ready")
print(f"     Model: {QA_MODEL}")
print("     Retrieval: TF-IDF cosine similarity over sentence chunks")


# ============================================================
# CELL 8: TRANSLATION - Helsinki-NLP OPUS-MT (MarianMT)
# ============================================================

from transformers import MarianMTModel, MarianTokenizer


def load_translation_model(lang_code: str) -> Tuple:
    """
    Lazily load MarianMT model for a specific target language.

    Each language model is cached independently in MODELS dict.
    Pattern: MODELS["translate_hi"] = (model, tokenizer)

    Why lazy per-language loading?
        - Each model is ~300 MB
        - Loading all 5 at once = 1.5 GB RAM (wasteful if user only needs 1)
        - Cache on first use -> instant on subsequent calls
    """
    key = f"translate_{lang_code}"
    if key not in MODELS:
        model_name = TRANSLATION_MODELS.get(lang_code)
        if not model_name:
            raise ValueError(
                f"Unsupported language: '{lang_code}'. "
                f"Use one of: {list(TRANSLATION_MODELS.keys())}"
            )
        print(f"[Translate] Loading {model_name}...")
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        MODELS[key] = (mdl, tok)
        lang_name = {v: k for k, v in LANG_MAP.items()}.get(lang_code, lang_code)
        print(f"[Translate] {lang_name} model loaded - cached as MODELS['{key}']")
    return MODELS[key]


def translate_chunk(text: str, model, tokenizer, max_len: int = 512) -> str:
    """Translate one chunk using MarianMT beam search (beam=4)."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    ).to(DEVICE)
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            num_beams=4,
            early_stopping=True,
            max_length=max_len,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def translate_text(text: str, target_lang: str) -> Dict[str, Any]:
    """
    Translate English text to the specified target language.

    Chunking Strategy (for long transcripts):
        1. Tokenize the text with the translation model tokenizer
        2. Split into chunks of at most 450 model tokens
        3. Translate each chunk independently
        4. Join translated chunks

    MarianMT generation settings:
        num_beams=4    : Beam search for quality translation
        max_length=512 : Match model context limit
        early_stopping : Stop beam search when EOS token generated

    Args:
        text        : English source text
        target_lang : ISO language code (hi/fr/de/es/ar)

    Returns dict:
        translated_text : Full translated string
        source_lang     : 'en'
        target_lang     : ISO code
        lang_name       : Full language name
        src_words       : Source word count
        tgt_words       : Translated word count
    """
    if not text.strip():
        raise ValueError("Provide English text to translate")
    model, tokenizer = load_translation_model(target_lang)
    lang_name = {v: k for k, v in LANG_MAP.items()}.get(target_lang, target_lang)

    print(f"[Translate] Target: {lang_name} | Input: {len(text.split())} words")

    # Token boundaries prevent silent truncation of long sentences or words.
    chunks = tokenizer_chunk(text, tokenizer, 450, 0)

    # Translate chunk by chunk
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"[Translate] Chunk {i+1}/{len(chunks)}...", end="\r")
        translated_chunks.append(translate_chunk(chunk, model, tokenizer))

    result_text = " ".join(translated_chunks)
    print(
        f"\n[Translate] Done: {len(chunks)} chunks -> {len(result_text.split())} words"
    )

    # Preserve names and technical terms, including Latin words in Hindi output.

    return {
        "translated_text": result_text,
        "source_lang": "en",
        "target_lang": target_lang,
        "lang_name": lang_name,
        "src_words": len(text.split()),
        "tgt_words": len(result_text.split()),
    }


print("[OK] Translation module ready")
print(f"     Supported: {', '.join(f'{k} ({v})' for k,v in LANG_MAP.items())}")
print("     Models load lazily on first use (~300 MB each)")


# ============================================================
# CELL 9: ACCURACY MEASUREMENT - WER + ROUGE + QA F1/EM
# ============================================================


def compute_wer_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Compute Word Error Rate and related metrics using jiwer.

    WER Formula:
        WER = (Substitutions + Deletions + Insertions) / N_reference_words

    Preprocessing transforms (ensures fair comparison):
        1. ToLowerCase          - case-insensitive comparison
        2. RemovePunctuation    - ignore punctuation differences
        3. RemoveMultipleSpaces - normalize whitespace
        4. Strip                - remove leading/trailing spaces
        5. ReduceToListOfListOfWords - tokenize for jiwer internals

    Additional metrics:
        MER (Match Error Rate) = (S+D) / max(|ref|, |hyp|)
        WIL (Word Info Lost)   = 1 - (WER/MER) * (1 - WER)
        Hits: correctly transcribed words

    Returns:
        Dict with wer, mer, wil, hits, substitutions, deletions, insertions
    """
    if not reference.strip():
        return {
            "wer": None,
            "mer": None,
            "wil": None,
            "hits": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
        }

    transforms = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
        ]
    )

    measures = jiwer.compute_measures(
        reference,
        hypothesis,
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )

    return {
        "wer": round(measures["wer"] * 100, 2),
        "mer": round(measures["mer"] * 100, 2),
        "wil": round(measures["wil"] * 100, 2),
        "hits": measures["hits"],
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
    }


def compute_rouge_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L using Google rouge-score.

    ROUGE-N Precision = |match_ngrams| / |hypothesis_ngrams|
    ROUGE-N Recall    = |match_ngrams| / |reference_ngrams|
    ROUGE-N F1        = 2 * P * R / (P + R)

    ROUGE-L uses Longest Common Subsequence (LCS) instead of n-grams,
    which measures sentence-level structural similarity.

    use_stemmer=True applies Porter stemming before comparison
    (so 'running' and 'run' count as matching).
    """
    if not reference.strip() or not hypothesis.strip():
        return {"rouge1_f1": None, "rouge2_f1": None, "rougeL_f1": None}

    scorer = rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    s = scorer.score(reference, hypothesis)

    return {
        "rouge1_precision": round(s["rouge1"].precision * 100, 2),
        "rouge1_recall": round(s["rouge1"].recall * 100, 2),
        "rouge1_f1": round(s["rouge1"].fmeasure * 100, 2),
        "rouge2_f1": round(s["rouge2"].fmeasure * 100, 2),
        "rougeL_f1": round(s["rougeL"].fmeasure * 100, 2),
    }


def badge(val, good_thr, fair_thr, invert=False) -> str:
    """Return quality badge: GREEN good, YELLOW fair, RED poor."""
    if val is None:
        return "[N/A]"
    if not invert:
        if val >= good_thr:
            return f"[GOOD]  {val}"
        if val >= fair_thr:
            return f"[FAIR]  {val}"
        return f"[POOR]  {val}"
    else:  # lower is better (e.g., WER)
        if val <= good_thr:
            return f"[GOOD]  {val}"
        if val <= fair_thr:
            return f"[FAIR]  {val}"
        return f"[POOR]  {val}"


def compute_accuracy_report(
    ref_transcript: str = "",
    hyp_transcript: str = "",
    ref_summary: str = "",
    hyp_summary: str = "",
    qa_refs: List[str] = None,
    qa_preds: List[str] = None,
) -> str:
    """
    Generate comprehensive accuracy report for all pipeline components.

    Args:
        ref_transcript : Ground truth transcript (human-verified)
        hyp_transcript : Model-generated transcript (faster-whisper output)
        ref_summary    : Reference summary (human-written or gold standard)
        hyp_summary    : Model-generated summary (BART output)
        qa_refs        : List of ground-truth answers
        qa_preds       : List of model-predicted answers

    Returns:
        Formatted multi-section accuracy report string
    """
    lines = [
        "=" * 65,
        "         STUDY MITRA - ACCURACY REPORT",
        "=" * 65,
    ]

    if (qa_refs is not None or qa_preds is not None) and (
        not qa_refs or not qa_preds or len(qa_refs) != len(qa_preds)
    ):
        raise ValueError("Provide matching QA references and predictions")

    # ── 1. Transcription WER ─────────────────────────────────
    lines += ["", "1. TRANSCRIPTION ACCURACY (Word Error Rate)", "-" * 65]
    if ref_transcript.strip():
        wer = compute_wer_metrics(ref_transcript, hyp_transcript)
        lines.append(
            f"   WER  : {badge(wer['wer'], 5, 10, invert=True)}%  (Target: < 10%)"
        )
        lines.append(f"   MER  : {wer['mer']}%")
        lines.append(f"   WIL  : {wer['wil']}%")
        lines.append(
            f"   Hits : {wer['hits']} | Subs: {wer['substitutions']} | Del: {wer['deletions']} | Ins: {wer['insertions']}"
        )
    else:
        lines.append(
            "   [N/A] Provide reference + hypothesis transcripts to compute WER"
        )

    # ── 2. Summarization ROUGE ───────────────────────────────
    lines += ["", "2. SUMMARIZATION QUALITY (ROUGE Scores)", "-" * 65]
    if ref_summary.strip() and hyp_summary.strip():
        rouge = compute_rouge_metrics(ref_summary, hyp_summary)
        lines.append(
            f"   ROUGE-1 F1 : {badge(rouge['rouge1_f1'], 40, 30)}  (>40 = Good)"
        )
        lines.append(
            f"   ROUGE-2 F1 : {badge(rouge['rouge2_f1'], 18, 10)}  (>18 = Good)"
        )
        lines.append(
            f"   ROUGE-L F1 : {badge(rouge['rougeL_f1'], 35, 25)}  (>35 = Good)"
        )

    else:
        lines.append(
            "   [N/A] Provide reference + hypothesis summaries to compute ROUGE"
        )

    # ── 3. QA F1 / EM ────────────────────────────────────────
    lines += ["", "3. QUESTION ANSWERING (F1 + Exact Match)", "-" * 65]
    if qa_refs and qa_preds and len(qa_refs) == len(qa_preds):
        qa_m = compute_qa_metrics(qa_preds, qa_refs)
        lines.append(
            f"   Exact Match : {badge(qa_m['exact_match'], 60, 40)}%  (>60 = Good)"
        )
        lines.append(f"   Token F1    : {badge(qa_m['f1'], 70, 50)}%  (>70 = Good)")
        lines.append(f"   Evaluated   : {qa_m['n_samples']} QA pairs")
    else:
        lines.append("   [N/A] Provide QA ground-truth pairs to compute F1/EM")

    lines.append("\nMetrics require reference data; no overall accuracy is inferred.")

    lines += [
        "",
        "=" * 65,
        "[GOOD] >= threshold | [FAIR] moderate | [POOR] below target | [N/A] not evaluated",
        "=" * 65,
    ]

    return "\n".join(lines)


print("[OK] Accuracy measurement module ready")
print("     WER    : jiwer with full preprocessing transforms")
print("     ROUGE  : rouge-score with Porter stemming")
print("     QA F1  : token-bag intersection (SQuAD evaluation style)")


# ============================================================
# CELL 9.5: GRADIO COMPATIBILITY PATCH
# Fixes: TypeError: argument of type 'bool' is not iterable
# Root cause: gr.File / gr.Slider emit additionalProperties:true
#             and gradio_client tries "const" in <bool>, which fails.
# Solution: Monkey-patch _json_schema_to_python_type to guard booleans.
# ============================================================

import gradio_client.utils as _gcu

_original_schema_fn = _gcu._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    """
    Patched version of gradio_client._json_schema_to_python_type.

    JSON Schema allows boolean schemas:
        true  -> accept any value
        false -> reject any value
    The original gradio_client code does not handle this case,
    causing a TypeError when it tries to do 'const' in <bool>.

    This patch returns 'Any' for boolean schemas (correct per JSON Schema spec).
    """
    if isinstance(schema, bool):
        return "Any"
    # Also guard dict schemas that have additionalProperties as a bool
    if isinstance(schema, dict):
        if "additionalProperties" in schema and isinstance(
            schema["additionalProperties"], bool
        ):
            schema = {**schema, "additionalProperties": {"type": "object"}}
    return _original_schema_fn(schema, defs)


# Apply the patch
_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type

print("[OK] Gradio bool-schema patch applied")
print("     gradio_client._json_schema_to_python_type now handles bool schemas safely")
print(
    "     This eliminates the 'TypeError: argument of type bool is not iterable' error"
)


# ============================================================
# CELL 10: GRADIO INTERFACE - 6-Tab AI Lecture Platform
# ============================================================

import gradio as gr
import tempfile, os


def run_transcribe(audio_file, model_choice, vad_choice, detect_lang):
    """
    Full pipeline: audio file -> FFmpeg extraction -> faster-whisper transcription.

    Args:
        audio_file  : Uploaded file (Gradio returns temp file path)
        model_choice: "large-v3 (GPU)" or "base (CPU)"
        vad_choice  : "VAD Enabled" or "VAD Disabled" (Radio, avoids Checkbox bool-schema bug)
        detect_lang : "Auto-detect" or specific language

    Returns:
        (full_transcript, timestamped_transcript, info_string)
    """
    if audio_file is None:
        return "Please upload an audio or video file.", "", "", ""

    wav_path = None
    try:
        # Step 1: Audio extraction
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        extract_audio(audio_file, wav_path)

        # Step 2: Model size selection
        size = (
            get_whisper_model_size()
            if model_choice == "Auto"
            else ("large-v3" if "large" in model_choice.lower() else "base")
        )

        # Step 3: Language
        lang = (
            None
            if detect_lang == "Auto-detect"
            else ({"English": "en", **LANG_MAP}[detect_lang])
        )

        # Step 4: VAD toggle (gr.Radio string -> bool)
        use_vad = vad_choice == "VAD Enabled"

        # Step 5: Transcription
        result = transcribe_audio(
            wav_path,
            model_size=size,
            language=lang,
            vad_filter=use_vad,
        )

        info = (
            f"Language: {result['language']} ({result['lang_prob']:.1%} confidence) | "
            f"Duration: {result['duration']:.1f}s | "
            f"Words: {result['num_words']} | "
            f"Segments: {result['num_segments']} | "
            f"RTF: {result['rtf']:.3f}"
        )

        return result["transcript"], result["timestamped"], info, result["transcript"]

    except Exception as e:
        return f"Error: {str(e)}", "", "", ""
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


# ─────────────────────────────────────────────────────────────
# TAB 2: SUMMARIZE
# ─────────────────────────────────────────────────────────────
def run_summarize(custom_text, detail_level, transcript=""):
    """Run BART summarization on cached transcript or custom text."""
    text = custom_text.strip() if custom_text.strip() else transcript

    if not text:
        return (
            "No transcript available. Please transcribe audio first (Tab 1) or paste text.",
            "",
        )

    try:
        result = summarize_text(text, detail=detail_level.lower())
        info = (
            f"Chunks processed: {result['num_chunks']} | "
            f"Input words: {result['input_words']} | "
            f"Output words: {result['output_words']} | "
            f"Compression: {result['compression_ratio']}x"
        )
        return result["summary"], info
    except Exception as e:
        return f"Error: {str(e)}", ""


# ─────────────────────────────────────────────────────────────
# TAB 3: KEYWORDS
# ─────────────────────────────────────────────────────────────
def run_keywords(custom_text, top_n, diversity, transcript=""):
    """Run KeyBERT + spaCy NER on cached transcript or custom text."""
    text = custom_text.strip() if custom_text.strip() else transcript

    if not text:
        return (
            "No transcript available. Please transcribe audio first (Tab 1) or paste text.",
            "",
        )

    try:
        result = extract_keywords_entities(
            text, top_n=int(top_n), diversity=float(diversity)
        )

        # Format keywords as table
        kw_lines = ["Rank  Keyword                         Score    Bar"]
        kw_lines.append("-" * 60)
        for i, kw in enumerate(result["keywords"], 1):
            bar = "#" * int(kw["score"] * 30)
            kw_lines.append(f"{i:<5} {kw['keyword']:<32} {kw['score']:.4f}  {bar}")
        kw_table = "\n".join(kw_lines)

        # Format entities
        ner_lines = ["\nNAMED ENTITIES:", "-" * 40]
        if result["entities"]:
            for label, ents in result["entities"].items():
                ner_lines.append(f"  {label:<15}: {', '.join(ents[:10])}")
        else:
            ner_lines.append("  No named entities found.")
        ner_text = "\n".join(ner_lines)

        return kw_table, ner_text
    except Exception as e:
        return f"Error: {str(e)}", ""


# ─────────────────────────────────────────────────────────────
# TAB 4: Q&A
# ─────────────────────────────────────────────────────────────
def run_qa(question, custom_context, top_k, transcript=""):
    """Run TF-IDF retrieval + RoBERTa QA on cached transcript or custom context."""
    context = custom_context.strip() if custom_context.strip() else transcript

    if not context:
        return (
            "No transcript available.",
            "Please transcribe audio first or paste context.",
            "",
        )

    if not question.strip():
        return "Please type a question.", "", ""

    try:
        result = answer_question(context, question, top_k=int(top_k))
        retrieved = "\n\n--- Retrieved Context ---\n" + "\n\n".join(
            [f"[Chunk {i+1}]\n{c}" for i, c in enumerate(result["chunks_used"])]
        )
        return result["answer"], f"Confidence: {result['confidence_pct']}", retrieved
    except Exception as e:
        return f"Error: {str(e)}", "", ""


# ─────────────────────────────────────────────────────────────
# TAB 5: TRANSLATE
# ─────────────────────────────────────────────────────────────
def run_translate(custom_text, target_language, transcript=""):
    """Translate cached transcript or custom text to selected language."""
    text = custom_text.strip() if custom_text.strip() else transcript

    if not text:
        return (
            "No transcript available. Please transcribe audio first (Tab 1) or paste text.",
            "",
        )

    try:
        lang_code = LANG_MAP.get(target_language, "fr")
        result = translate_text(text, lang_code)
        info = (
            f"Language: English -> {result['lang_name']} | "
            f"Source words: {result['src_words']} | "
            f"Translated words: {result['tgt_words']}"
        )
        return result["translated_text"], info
    except Exception as e:
        return f"Error: {str(e)}", ""


# ─────────────────────────────────────────────────────────────
# TAB 6: ACCURACY
# ─────────────────────────────────────────────────────────────
def run_accuracy(
    ref_transcript,
    hyp_transcript,
    ref_summary,
    hyp_summary,
    qa_gt,
    qa_pred_str,
    transcript="",
):
    """Compute WER, ROUGE, QA F1/EM accuracy report."""
    # Parse QA pairs from newline-separated input
    qa_refs = (
        [x.strip() for x in qa_gt.strip().splitlines() if x.strip()] if qa_gt else None
    )
    qa_preds = (
        [x.strip() for x in qa_pred_str.strip().splitlines() if x.strip()]
        if qa_pred_str
        else None
    )

    if qa_refs and qa_preds and len(qa_refs) != len(qa_preds):
        return "ERROR: Number of QA ground truths must match number of predictions (one per line)."

    try:
        report = compute_accuracy_report(
            ref_transcript=ref_transcript,
            hyp_transcript=hyp_transcript if hyp_transcript.strip() else transcript,
            ref_summary=ref_summary,
            hyp_summary=hyp_summary,
            qa_refs=qa_refs,
            qa_preds=qa_preds,
        )
        return report
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# BUILD GRADIO INTERFACE
# =============================================================================


def build_app():
    with gr.Blocks(
        title="Study Mitra - AI Lecture Intelligence",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
            .header-box { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3a5f, #2980b9); border-radius: 12px; margin-bottom: 16px; }
            .header-box h1 { color: white; font-size: 2rem; margin: 0; }
            .header-box p  { color: #cce4f7; margin: 4px 0 0 0; }
            .metric-box { background: #f0f7ff; border-left: 4px solid #2980b9; padding: 8px 12px; border-radius: 6px; font-family: monospace; }
            footer { display: none !important; }
        """,
    ) as demo:

        transcript_state = gr.State("")

        # ── Header ───────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <h1>STUDY MITRA</h1>
            <p>AI-Powered Lecture Intelligence Platform &nbsp;|&nbsp; faster-whisper &bull; BART &bull; KeyBERT &bull; RoBERTa &bull; Helsinki-NLP</p>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: TRANSCRIBE ─────────────────────────────────
            with gr.TabItem("Transcribe"):
                gr.Markdown(
                    "### Upload audio or video to generate transcript with timestamps"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t1_file = gr.File(
                            label="Upload File (.mp4, .mkv, .mp3, .wav, .webm)",
                            file_types=[
                                ".mp4",
                                ".mkv",
                                ".webm",
                                ".mp3",
                                ".wav",
                                ".flac",
                                ".m4a",
                                ".ogg",
                            ],
                        )
                        t1_model = gr.Radio(
                            ["Auto", "large-v3 (GPU - Best)", "base (CPU - Fast)"],
                            value="Auto",
                            label="Whisper Model",
                        )
                        t1_vad = gr.Radio(
                            ["VAD Enabled", "VAD Disabled"],
                            value="VAD Enabled",
                            label="Voice Activity Detection (removes silence, improves WER)",
                        )
                        t1_lang = gr.Dropdown(
                            [
                                "Auto-detect",
                                "English",
                                "Hindi",
                                "French",
                                "German",
                                "Spanish",
                                "Arabic",
                            ],
                            value="Auto-detect",
                            label="Force Language (optional)",
                        )
                        t1_btn = gr.Button("Transcribe", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        t1_out_plain = gr.Textbox(
                            label="Full Transcript", lines=12, show_copy_button=True
                        )
                        t1_out_ts = gr.Textbox(
                            label="Timestamped Transcript",
                            lines=8,
                            show_copy_button=True,
                        )
                        t1_info = gr.Textbox(
                            label="Info",
                            lines=2,
                            interactive=False,
                            elem_classes=["metric-box"],
                        )

                t1_btn.click(
                    run_transcribe,
                    inputs=[t1_file, t1_model, t1_vad, t1_lang],
                    outputs=[t1_out_plain, t1_out_ts, t1_info, transcript_state],
                )

            # ── TAB 2: SUMMARIZE ──────────────────────────────────
            with gr.TabItem("Summarize"):
                gr.Markdown(
                    "### Generate abstractive summary using BART-large-CNN (Map-Reduce)"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t2_text = gr.Textbox(
                            label="Custom Text (leave blank to use transcript)",
                            lines=6,
                            placeholder="Paste text here, or leave blank to use Tab 1 transcript...",
                        )
                        t2_detail = gr.Radio(
                            ["Brief", "Medium", "Detailed"],
                            value="Medium",
                            label="Detail Level",
                        )
                        t2_btn = gr.Button("Summarize", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        t2_out = gr.Textbox(
                            label="Summary", lines=12, show_copy_button=True
                        )
                        t2_info = gr.Textbox(
                            label="Compression Info",
                            lines=2,
                            interactive=False,
                            elem_classes=["metric-box"],
                        )

                t2_btn.click(
                    run_summarize,
                    inputs=[t2_text, t2_detail, transcript_state],
                    outputs=[t2_out, t2_info],
                )

            # ── TAB 3: KEYWORDS ───────────────────────────────────
            with gr.TabItem("Keywords"):
                gr.Markdown(
                    "### Extract semantic keywords (KeyBERT + MMR) and named entities (spaCy NER)"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_text = gr.Textbox(
                            label="Custom Text (leave blank for transcript)", lines=6
                        )
                        t3_top_n = gr.Slider(
                            5, 25, value=15, step=1, label="Top N Keywords"
                        )
                        t3_diversity = gr.Slider(
                            0.0,
                            1.0,
                            value=0.7,
                            step=0.1,
                            label="MMR Diversity (0=redundant, 1=diverse)",
                        )
                        t3_btn = gr.Button(
                            "Extract Keywords", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        t3_kw = gr.Textbox(
                            label="Keywords (ranked by BERT cosine similarity)",
                            lines=12,
                            show_copy_button=True,
                            elem_classes=["metric-box"],
                        )
                        t3_ner = gr.Textbox(
                            label="Named Entities (spaCy)",
                            lines=8,
                            show_copy_button=True,
                        )

                t3_btn.click(
                    run_keywords,
                    inputs=[t3_text, t3_top_n, t3_diversity, transcript_state],
                    outputs=[t3_kw, t3_ner],
                )

            # ── TAB 4: Q&A ────────────────────────────────────────
            with gr.TabItem("Q&A"):
                gr.Markdown(
                    "### Ask questions answered by RoBERTa-SQuAD2 with TF-IDF context retrieval"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t4_question = gr.Textbox(
                            label="Your Question",
                            lines=3,
                            placeholder="E.g. What is the main topic? Who is mentioned?",
                        )
                        t4_ctx = gr.Textbox(
                            label="Custom Context (leave blank for transcript)", lines=5
                        )
                        t4_top_k = gr.Slider(
                            1, 5, value=3, step=1, label="Top-K TF-IDF Chunks"
                        )
                        t4_btn = gr.Button(
                            "Answer Question", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        t4_answer = gr.Textbox(
                            label="Answer", lines=3, show_copy_button=True
                        )
                        t4_confidence = gr.Textbox(
                            label="Confidence Score",
                            lines=1,
                            interactive=False,
                            elem_classes=["metric-box"],
                        )
                        t4_ctx_out = gr.Textbox(
                            label="Retrieved Context (TF-IDF)", lines=10
                        )

                t4_btn.click(
                    run_qa,
                    inputs=[t4_question, t4_ctx, t4_top_k, transcript_state],
                    outputs=[t4_answer, t4_confidence, t4_ctx_out],
                )

            # ── TAB 5: TRANSLATE ──────────────────────────────────
            with gr.TabItem("Translate"):
                gr.Markdown(
                    "### Translate to 5 languages using Helsinki-NLP OPUS-MT (offline, free)"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_text = gr.Textbox(
                            label="Text to Translate (leave blank for transcript)",
                            lines=8,
                            placeholder="Paste English text here or leave blank...",
                        )
                        t5_lang = gr.Dropdown(
                            list(LANG_MAP.keys()),
                            value="Hindi",
                            label="Target Language",
                        )
                        t5_btn = gr.Button("Translate", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        t5_out = gr.Textbox(
                            label="Translation", lines=12, show_copy_button=True
                        )
                        t5_info = gr.Textbox(
                            label="Translation Info",
                            lines=2,
                            interactive=False,
                            elem_classes=["metric-box"],
                        )

                t5_btn.click(
                    run_translate,
                    inputs=[t5_text, t5_lang, transcript_state],
                    outputs=[t5_out, t5_info],
                )

            # ── TAB 6: ACCURACY ───────────────────────────────────
            with gr.TabItem("Accuracy / Evaluation"):
                gr.Markdown("### Evaluate against human-verified references")
                gr.Markdown(
                    "WER, ROUGE and QA F1/Exact Match require reference data. Blank sections are not evaluated."
                )
                ref_t = gr.Textbox(label="Reference transcript", lines=4)
                hyp_t = gr.Textbox(
                    label="Predicted transcript (blank uses this session)", lines=4
                )
                ref_s = gr.Textbox(label="Reference summary", lines=3)
                hyp_s = gr.Textbox(label="Predicted summary", lines=3)
                qa_gt = gr.Textbox(label="Reference answers (one per line)", lines=3)
                qa_pred = gr.Textbox(label="Predicted answers (one per line)", lines=3)
                t6_btn = gr.Button("Compute Reference Metrics", variant="primary")
                t6_report = gr.Textbox(label="Evaluation Report", lines=20)
                t6_btn.click(
                    run_accuracy,
                    inputs=[
                        ref_t,
                        hyp_t,
                        ref_s,
                        hyp_s,
                        qa_gt,
                        qa_pred,
                        transcript_state,
                    ],
                    outputs=[t6_report],
                )

        gr.Markdown("""
        ---
        **Study Mitra** | Powered by faster-whisper, BART, KeyBERT, RoBERTa, Helsinki-NLP
        *Models run locally on CPU or CUDA. Downloads require internet on first use.*
        """)
    return demo
