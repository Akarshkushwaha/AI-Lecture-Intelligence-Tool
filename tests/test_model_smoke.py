"""Opt-in real ASR check; downloads the tiny English model and Whisper test audio."""

import os
from pathlib import Path
import tempfile
import unittest
from urllib.request import urlretrieve

import study_mitra as sm


@unittest.skipUnless(
    os.environ.get("STUDYMITRA_MODEL_SMOKE") == "1",
    "Set STUDYMITRA_MODEL_SMOKE=1 to download and test real model weights",
)
class ModelSmokeTests(unittest.TestCase):
    def test_real_english_transcription(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jfk.flac"
            # The reference fixture used by the upstream Whisper project.
            urlretrieve(
                "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
                path,
            )
            result = sm.transcribe_audio(
                str(path),
                model_size="tiny.en",
                language="en",
                vad_filter=False,
                word_timestamps=True,
            )
            self.assertIn("country", result["transcript"].lower())
            self.assertIn("ask", result["transcript"].lower())
            self.assertGreater(result["duration"], 0)
            self.assertTrue(result["segments"])
