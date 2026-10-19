"""Regression checks without downloading model weights."""

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import study_mitra as sm


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


class RegressionTests(unittest.TestCase):
    def test_metrics_keep_order_and_repeated_tokens(self):
        self.assertEqual(
            sm.compute_qa_metrics(["red blue"], ["blue red"])["exact_match"], 0
        )
        self.assertEqual(
            sm.compute_qa_metrics(["red red blue"], ["red blue"])["f1"], 80
        )
        self.assertEqual(
            sm.compute_qa_metrics(["The cat!"], ["cat"])["exact_match"], 100
        )
        self.assertEqual(sm.compute_qa_metrics([""], [""])["f1"], 100)
        with self.assertRaises(ValueError):
            sm.compute_qa_metrics(["one"], ["one", "two"])

    def test_real_reference_metrics(self):
        self.assertEqual(sm.compute_wer_metrics("one two", "one three")["wer"], 50)
        self.assertEqual(sm.compute_wer_metrics("one two", "")["wer"], 100)
        self.assertEqual(
            sm.compute_rouge_metrics("red blue", "red blue")["rouge1_f1"], 100
        )
        report = sm.compute_accuracy_report(
            ref_transcript="one two", hyp_transcript="one three"
        )
        self.assertIn("50.0", report)
        self.assertNotIn("OVERALL SYSTEM ACCURACY", report)
        self.assertIn("[N/A]", sm.compute_accuracy_report())
        with self.assertRaises(ValueError):
            sm.compute_accuracy_report(qa_refs=["a"], qa_preds=None)

    def test_chunk_bounds(self):
        tok = WordTokenizer()
        self.assertEqual(sm.tokenizer_chunk("a b c d e", tok, 3, 1), ["a b c", "c d e"])
        with self.assertRaises(ValueError):
            sm.tokenizer_chunk("a b", tok, 3, 3)

    def test_summary_reduce_covers_late_lecture(self):
        tok = WordTokenizer()
        text = " ".join(["early"] * 8000 + ["FINAL_TOPIC"])
        seen = []

        def summarize(chunk, *args):
            seen.append(chunk)
            # Compress each chunk while retaining its ending topic.
            return " ".join([chunk.split()[-1]] * 180)

        with patch.object(sm, "load_bart", return_value=(None, tok)), patch.object(
            sm, "summarize_one_chunk", side_effect=summarize
        ):
            result = sm.summarize_text(text)
        self.assertIn("FINAL_TOPIC", result["summary"])
        self.assertTrue(all(len(c.split()) <= sm.MAX_CHUNK_TOKENS for c in seen))

    def test_summary_nonshrinking_model_terminates(self):
        with patch.object(
            sm, "load_bart", return_value=(None, WordTokenizer())
        ), patch.object(
            sm, "summarize_one_chunk", side_effect=lambda text, *args: text
        ):
            with self.assertRaises(RuntimeError):
                sm.summarize_text("word " * 3000)

    def test_qa_preserves_late_context_and_no_answer(self):
        text = "word " * 600 + "FINAL_ANSWER"
        with patch.object(
            sm, "retrieve_context_tfidf", return_value=(text, [text])
        ), patch.object(sm, "load_qa_pipeline") as load:
            load.return_value.return_value = {"score": 0.95, "answer": "FINAL_ANSWER"}
            result = sm.answer_question(text, "What?")
            self.assertIn("FINAL_ANSWER", load.return_value.call_args.kwargs["context"])
            self.assertEqual(result["answer"], "FINAL_ANSWER")
            load.return_value.return_value = {"score": 0.95, "answer": ""}
            self.assertIn("not found", sm.answer_question(text, "What?")["answer"])

    def test_translation_preserves_technical_terms_and_tail(self):
        tok = WordTokenizer()
        with patch.object(
            sm, "load_translation_model", return_value=(None, tok)
        ), patch.object(
            sm, "translate_chunk", side_effect=lambda text, *args: text
        ) as translate:
            text = "Python " * 1000 + "FINAL_TERM"
            result = sm.translate_text(text, "hi")
            self.assertEqual(result["translated_text"], text.strip())
            self.assertTrue(
                all(len(c.args[0].split()) <= 450 for c in translate.call_args_list)
            )

    def test_language_mapping_cleanup_and_failure_state(self):
        paths = []

        def extract(source, output):
            paths.append(output)

        result = {
            "transcript": "lecture",
            "timestamped": "[00:00] lecture",
            "language": "de",
            "lang_prob": 0.9,
            "duration": 2,
            "num_words": 1,
            "num_segments": 1,
            "rtf": 0.1,
        }
        with patch.object(sm, "extract_audio", side_effect=extract), patch.object(
            sm, "transcribe_audio", return_value=result
        ) as transcribe:
            for name, code in {"English": "en", **sm.LANG_MAP}.items():
                output = sm.run_transcribe("upload.mp4", "Auto", "VAD Enabled", name)
                self.assertEqual(output[3], "lecture")
                self.assertEqual(transcribe.call_args.kwargs["language"], code)
            transcribe.side_effect = RuntimeError("failed")
            self.assertEqual(
                sm.run_transcribe("upload.mp4", "Auto", "VAD Enabled", "English")[3], ""
            )
        self.assertTrue(all(not Path(p).exists() for p in paths))
        self.assertEqual(
            len(sm.run_transcribe(None, "Auto", "VAD Enabled", "English")), 4
        )

    def test_sessions_use_their_own_transcript(self):
        def summarize(text, **kwargs):
            return {
                "summary": text,
                "num_chunks": 1,
                "input_words": 1,
                "output_words": 1,
                "compression_ratio": 1,
            }

        with patch.object(sm, "summarize_text", side_effect=summarize):
            self.assertEqual(sm.run_summarize("", "Brief", "alice")[0], "alice")
            self.assertEqual(sm.run_summarize("", "Brief", "bob")[0], "bob")
            self.assertIn("No transcript", sm.run_summarize("", "Brief")[0])

    def test_ui_schema_and_session_wiring(self):
        demo = sm.build_app()
        demo.get_api_info()
        state_ids = [i for i, b in demo.blocks.items() if isinstance(b, sm.gr.State)]
        self.assertEqual(len(state_ids), 1)
        deps = demo.config["dependencies"]
        self.assertIn(state_ids[0], deps[0]["outputs"])
        for dep in deps[1:]:
            self.assertIn(state_ids[0], dep["inputs"])

    def test_moviepy2_audio_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "input.wav", Path(folder) / "output.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 16000)
            with patch.object(sm, "check_ffmpeg", return_value=False):
                sm.extract_audio(str(source), str(output))
            with wave.open(str(output)) as audio:
                self.assertEqual(audio.getframerate(), 16000)
                self.assertEqual(audio.getnchannels(), 1)


if __name__ == "__main__":
    unittest.main()
