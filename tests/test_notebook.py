"""The teaching notebook must retain the full, runnable implementation."""

import ast
import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

from sync_notebook import SETUP_HELPERS, build_notebook


class NotebookTests(unittest.TestCase):
    def test_setup_forwards_stdout_and_stderr(self):
        namespace = {"subprocess": subprocess}
        exec(SETUP_HELPERS, namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            namespace["run_setup"]("Test install", [sys.executable, "-u", "-c",
                "import sys; print('download progress'); print('warning text', file=sys.stderr)"])
        self.assertIn("download progress", output.getvalue())
        self.assertIn("warning text", output.getvalue())
        self.assertIn("[OK] Test install", output.getvalue())

    def test_setup_preserves_errors_without_success_message(self):
        namespace = {"subprocess": subprocess}
        exec(SETUP_HELPERS, namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError):
            namespace["run_setup"]("Test failure", [sys.executable, "-u", "-c",
                "import sys; print('installation failed', flush=True); sys.exit(7)"])
        self.assertIn("installation failed", output.getvalue())
        self.assertNotIn("[OK]", output.getvalue())

    def test_notebook_contains_full_current_implementation(self):
        path = Path(__file__).resolve().parents[1] / "lecture-intelligence-tool.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook, build_notebook())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code), 12)
        for source in code:
            ast.parse(source)
        implementation = "".join(code[1:-1])
        module = path.with_name("study_mitra.py").read_text(encoding="utf-8")
        self.assertEqual(implementation, module)
        self.assertNotIn("from study_mitra import", "".join(code))


if __name__ == "__main__":
    unittest.main()
