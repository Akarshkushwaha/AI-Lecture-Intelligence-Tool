"""The teaching notebook must retain the full, runnable implementation."""

import ast
import json
import unittest
from pathlib import Path

from sync_notebook import build_notebook


class NotebookTests(unittest.TestCase):
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
