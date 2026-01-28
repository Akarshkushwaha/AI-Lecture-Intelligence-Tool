import gradio as gr
import os
import torch
import spacy
import re
import random
import math
import yt_dlp
import nltk
import nltk.data
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
from faster_whisper import WhisperModel

# --- 1. SAFE DEPENDENCY LOADING ---
# NLTK Data Check
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("⬇️ Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Spacy Model Check
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 AI Core initialized on {DEVICE}")

# --- 2. AI ENGINE (MODELS) ---
class AI_Engine:
    def __init__(self):
        print("⏳ Initializing AI Models...")

        # 1. WHISPER (Audio to Text) - Safe Loading
        try:
            self.whisper = WhisperModel("medium", device="cuda", compute_type="float16")
            print("   ✅ Whisper loaded on CUDA")
        except Exception:
            print("   ⚠️ CUDA failed. Switching to CPU Mode...")
            self.whisper = WhisperModel("medium", device="cpu", compute_type="int8")

        # 2. SUMMARIZER (BART)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

        # 3. TRANSLATOR (M2M100)
        self.trans_model_name = "facebook/m2m100_418M"
        self.trans_tokenizer = M2M100Tokenizer.from_pretrained(self.trans_model_name)
        self.trans_model = M2M100ForConditionalGeneration.from_pretrained(self.trans_model_name)
        if torch.cuda.is_available():
            self.trans_model = self.trans_model.to("cuda")

        # 4. QUIZ GENERATOR (T5)
        self.qg_pipe = pipeline('text2text-generation', model='valhalla/t5-small-qa-qg-hl', device=0 if torch.cuda.is_available() else -1)

engine = AI_Engine()

# --- 3. ACCURACY CALCULATION LOGIC (ON DEMAND) ---
def calculate_metrics(segments_data, transcript_text, summary_text, notes_text):
    """
    Ye function tabhi chalega jab user 'Calculate Accuracy' button dabayega.
    """
    print("📊 Calculating Accuracy Metrics...")

    # 1. Transcription Accuracy (Confidence Score)
    trans_score = 0.0
    if segments_data:
        total_prob = 0
        count = 0
        for seg in segments_data:
            # Avg logprob ko percentage mein badalna
            prob = math.exp(seg.avg_logprob)
            total_prob += prob
            count += 1
        trans_score = round((total_prob / count) * 100, 1) if count > 0 else 0.0

    # Helper for Text Similarity (Jaccard Index)
    def get_text_overlap(source, target):
        if not source or not target: return 0.0
        stop_words = set(nltk.corpus.stopwords.words('english'))

        def tokenize(txt):
            return set([t.lower() for t in nltk.word_tokenize(txt) if t.isalnum() and t not in stop_words])

        src_tokens = tokenize(source)
        tgt_tokens = tokenize(target)

        if not tgt_tokens: return 0.0

        # Check faithfulness: kitne target words source mein maujood hain
        intersection = src_tokens.intersection(tgt_tokens)
        return round((len(intersection) / len(tgt_tokens)) * 100, 1)

    # 2. Summary Accuracy
    summ_score = get_text_overlap(transcript_text, summary_text)

    # 3. Notes Accuracy
    notes_score = get_text_overlap(transcript_text, notes_text)

    return trans_score, summ_score, notes_score

# --- 4. PROCESSING FUNCTIONS ---

def download_video(url, progress=gr.Progress()):
    if not url: return None, "Please enter a URL."
    progress(0, desc="Downloading...")
    output_path = "downloads"
    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename, f"✅ Video downloaded: {info.get('title')}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def process_transcription(video_path, progress=gr.Progress()):
    if not video_path: return None, None, "Please download a video first."

    progress(0.2, desc="Transcribing...")
    try:
        segments_generator, info = engine.whisper.transcribe(video_path, beam_size=5)
        # Convert to list to store in State for later accuracy calc
        segments = list(segments_generator)
        transcript_text = " ".join([seg.text for seg in segments]).strip()

        return transcript_text, segments, "✅ Transcription Complete"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def generate_summary(text):
    if not text: return "No transcript."

    # Smart Chunking logic (Sentence based)
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) < 2500:
            current_chunk += sent + " "
        else:
            chunks.append(current_chunk)
            current_chunk = sent + " "
    if current_chunk: chunks.append(current_chunk)

    summaries = []
    for chunk in chunks[:3]:
        try:
            input_len = len(chunk.split())
            max_l = min(150, int(input_len * 0.6))
            min_l = min(30, int(input_len * 0.2))
            if max_l > min_l:
                summary = engine.summarizer(chunk, max_length=max_l, min_length=min_l, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        except: pass

    return " ".join(summaries)

def generate_notes(text):
    if not text: return "No transcript."

    sentences = nltk.tokenize.sent_tokenize(text)
    chunk_size = 7
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

    notes = []
    for chunk in chunks[:5]:
        try:
            res = engine.summarizer(chunk[:2500], max_length=60, min_length=15, do_sample=False)
            point = res[0]['summary_text']
            notes.append(f"- {point}")
        except: pass

    return "### Key Takeaways:\n" + "\n".join(notes)

def generate_quiz(text):
    if not text: return "No transcript."
    doc = nlp(text[:5000])
    entities = [ent.text for ent in doc.ents if ent.label_ in ['DATE', 'ORG', 'PERSON', 'GPE', 'EVENT']]
    unique_entities = list(set(entities))

    if len(unique_entities) < 3:
        unique_entities = [token.text for token in doc if token.pos_ == "NOUN" and len(token.text) > 4]
        unique_entities = list(set(unique_entities))

    random.shuffle(unique_entities)
    selected_answers = unique_entities[:5]
    quiz_output = ""

    for ans in selected_answers:
        for sent in nltk.tokenize.sent_tokenize(text):
            if ans in sent and len(sent) < 200:
                pattern = re.compile(re.escape(ans), re.IGNORECASE)
                input_text = pattern.sub(f"<hl>{ans}<hl>", sent)
                try:
                    q = engine.qg_pipe(f"generate question: {input_text}")[0]['generated_text']
                    quiz_output += f"**Q:** {q}\n**A:** ||{ans}||\n\n"
                    break
                except: continue
    return quiz_output

def translate_text(text, target_lang):
    if not text: return "No transcript."
    lang_map = {"Hindi": "hi", "French": "fr", "English": "en"}
    code = lang_map.get(target_lang, "en")
    tokenizer = engine.trans_tokenizer
    model = engine.trans_model
    tokenizer.src_lang = "en"
    encoded = tokenizer(text[:600], return_tensors="pt").to(DEVICE)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(code))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# --- 5. UI SETUP ---
custom_css = """
.gradio-container { background: linear-gradient(to bottom right, #0F172A, #1E1B4B) !important; }
body, .prose, .markdown-text, label, span, p, h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; font-family: 'Inter', sans-serif; }
.markdown-text, .prose p, .prose h1, .prose h2, .prose h3, .prose li, label span { margin-left: 12px !important; }
textarea, input, .gr-box, .prose, #output_box, #status_box { background-color: rgba(30, 41, 59, 0.8) !important; color: #FFFFFF !important; border: 1px solid #334155 !important; border-radius: 12px !important; backdrop-filter: blur(5px); }
h1 { background: linear-gradient(90deg, #818CF8, #22D3EE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-weight: 800 !important; margin-bottom: 10px !important; margin-left: 0 !important; }
button.primary { background: linear-gradient(90deg, #4F46E5, #7C3AED) !important; color: white !important; border: none !important; }
button.secondary { background-color: #334155 !important; color: #E2E8F0 !important; border: 1px solid #475569 !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Team Racoon Tool") as demo:

    # STATE VARIABLES (To store data for on-demand accuracy check)
    video_path_state = gr.State()
    transcript_state = gr.State()
    segments_state = gr.State()
    summary_state = gr.State()
    notes_state = gr.State()

    # --- HEADER ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 🎓 AI Powered Lecture Intelligence Tool")

    # --- INPUT ---
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown('### 1. Source & Status')
            url_input = gr.Textbox(label="YouTube URL", placeholder="Paste video link...", lines=1)
            download_btn = gr.Button("⬇️ Load Video", variant="primary", size="lg")
            status_msg = gr.Textbox(label="System Logs", value="System Ready...", interactive=False, lines=6, elem_id="status_box")

        with gr.Column(scale=1):
            gr.Markdown('### 2. Video Preview')
            video_player = gr.Video(label="Player", interactive=True, elem_id="video_player")

    gr.Markdown("---")

    # --- OUTPUT ---
    with gr.Row(variant="panel"):

        # CONTROLS
        with gr.Column(scale=1):
            gr.Markdown("### 3. Controls")
            analyze_btn = gr.Button("✨ Transcribe", variant="primary", size="lg", interactive=False)

            gr.Markdown("#### Generation Tools")
            summ_btn = gr.Button("📝 Generate Summary", variant="secondary")
            notes_btn = gr.Button("📌 Extract Notes", variant="secondary")
            quiz_btn = gr.Button("❓ Create Quiz", variant="secondary")

            gr.Markdown("#### Translation")
            with gr.Row():
                lang_select = gr.Dropdown(["Hindi", "French", "English"], label="Target Language", value="Hindi")
                trans_btn = gr.Button("GO", variant="secondary")

        # RESULTS TAB
        with gr.Column(scale=2):
            gr.Markdown("### 4. Intelligence Output")
            with gr.Tabs():
                with gr.TabItem("📄 Transcript"):
                    transcript_output = gr.Textbox(label="Full Text", lines=15, show_copy_button=True, elem_id="output_box")
                with gr.TabItem("📝 Summary"):
                    summary_output = gr.Textbox(label="Abstract", lines=10, show_copy_button=True, elem_id="output_box")
                with gr.TabItem("📌 Notes"):
                    notes_output = gr.Markdown(elem_id="output_box")
                with gr.TabItem("❓ Quiz"):
                    quiz_output = gr.Markdown(elem_id="output_box")
                with gr.TabItem("🌍 Translate"):
                    trans_output = gr.Textbox(label="Translation", lines=10, elem_id="output_box")

    # --- ACCURACY SECTION (ON DEMAND) ---
    gr.Markdown("---")
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            acc_btn = gr.Button("📊 Calculate Accuracy Scores", variant="primary", size="lg")
            gr.Markdown("*Generate Summary & Notes first.*")

        with gr.Column(scale=3):
            with gr.Row():
                acc_trans_out = gr.Number(label="Transcription Confidence %", value=0)
                acc_summ_out = gr.Number(label="Summary Fidelity %", value=0)
                acc_notes_out = gr.Number(label="Notes Relevance %", value=0)

    # --- EVENTS ---

    # 1. Download
    def on_download(url):
        path, msg = download_video(url)
        return path, path, msg, gr.update(interactive=(path is not None))

    download_btn.click(on_download, inputs=[url_input], outputs=[video_player, video_path_state, status_msg, analyze_btn])

    # 2. Transcribe (Saves segments to State)
    def on_transcribe(video_path):
        text, segments, msg = process_transcription(video_path)
        return text, text, segments, msg

    analyze_btn.click(on_transcribe,
                      inputs=[video_path_state],
                      outputs=[transcript_output, transcript_state, segments_state, status_msg])

    # 3. Summary (Saves summary to State)
    def on_summary(text):
        summ = generate_summary(text)
        return summ, summ

    summ_btn.click(on_summary, inputs=[transcript_state], outputs=[summary_output, summary_state])

    # 4. Notes (Saves notes to State)
    def on_notes(text):
        notes = generate_notes(text)
        return notes, notes

    notes_btn.click(on_notes, inputs=[transcript_state], outputs=[notes_output, notes_state])

    # 5. Quiz & Translate
    quiz_btn.click(generate_quiz, inputs=[transcript_state], outputs=[quiz_output])
    trans_btn.click(translate_text, inputs=[transcript_state, lang_select], outputs=[trans_output])

    # 6. ACCURACY CALCULATION (Separate Trigger)
    acc_btn.click(calculate_metrics,
                  inputs=[segments_state, transcript_state, summary_state, notes_state],
                  outputs=[acc_trans_out, acc_summ_out, acc_notes_out])

demo.queue().launch(share=True, debug=True)
