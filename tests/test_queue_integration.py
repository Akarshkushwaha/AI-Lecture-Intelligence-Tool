"""Exercise real uploads and queued callbacks without downloading model weights."""

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from gradio_client import Client, handle_file

import study_mitra as sm


class QueueIntegrationTests(unittest.TestCase):
    def test_upload_transcribe_and_session_reuse_through_queue(self):
        result = {
            "transcript": "A lecture about gravity.",
            "timestamped": "[00:00] A lecture about gravity.",
            "language": "en", "lang_prob": .99, "duration": 1,
            "num_words": 5, "num_segments": 1, "rtf": .1,
        }

        def summarize(text, **kwargs):
            return {"summary": text, "num_chunks": 1, "input_words": 5,
                    "output_words": 5, "compression_ratio": 1}

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "lecture.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 16000)
            app = sm.build_app().queue(max_size=5, default_concurrency_limit=1)
            clients = []
            try:
                _, url, _ = app.launch(share=False, prevent_thread_lock=True,
                                       quiet=True, show_error=True)
                with patch.object(sm, "transcribe_audio", return_value=result), patch.object(
                    sm, "summarize_text", side_effect=summarize
                ):
                    client = Client(url, verbose=False)
                    clients.append(client)
                    transcript = client.submit(handle_file(str(source)), "Auto",
                                               "VAD Enabled", "Auto-detect",
                                               api_name="/run_transcribe").result(timeout=30)
                    self.assertEqual(transcript[0], result["transcript"])
                    summary = client.submit("", "Brief", api_name="/run_summarize").result(timeout=30)
                    self.assertEqual(summary[0], result["transcript"])
                    other = Client(url, verbose=False)
                    clients.append(other)
                    empty = other.submit("", "Brief", api_name="/run_summarize").result(timeout=30)
                    self.assertIn("No transcript", empty[0])
            finally:
                for client in clients:
                    client.close()
                app.close()


if __name__ == "__main__":
    unittest.main()
