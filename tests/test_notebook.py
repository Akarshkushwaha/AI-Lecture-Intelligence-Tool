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
    def test_linux_lock_satisfies_declared_dependencies(self):
        from packaging.requirements import Requirement

        root = Path(__file__).resolve().parents[1]
        locked = {}
        for line in (root / "requirements-colab.lock").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                requirement = Requirement(line)
                locked[requirement.name.replace("_", "-")] = requirement
        for line in (root / "requirements.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith("https://"):
                self.assertIn(line, [r.url for r in locked.values()])
                continue
            requirement = Requirement(line)
            pin = locked[requirement.name.replace("_", "-")]
            version = next(iter(pin.specifier)).version
            self.assertIn(version, requirement.specifier, line)

    def test_setup_forwards_stdout_and_stderr(self):
        namespace = {}
        exec(SETUP_HELPERS, namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            namespace["run_setup"](
                "Test install",
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys; print('download progress'); print('warning text', file=sys.stderr)",
                ],
            )
        self.assertIn("download progress", output.getvalue())
        self.assertIn("warning text", output.getvalue())
        self.assertIn("[OK] Test install", output.getvalue())

    def test_setup_preserves_errors_without_success_message(self):
        namespace = {}
        exec(SETUP_HELPERS, namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(
            subprocess.CalledProcessError
        ):
            namespace["run_setup"](
                "Test failure",
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys; print('installation failed', flush=True); sys.exit(7)",
                ],
            )
        self.assertIn("installation failed", output.getvalue())
        self.assertNotIn("[OK]", output.getvalue())

    def test_notebook_contains_full_current_implementation(self):
        path = Path(__file__).resolve().parents[1] / "lecture-intelligence-tool.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook, build_notebook())
        code = [
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        ]
        self.assertEqual(len(code), 12)
        for source in code:
            ast.parse(
                source.split("\n", 1)[1]
                if source.startswith("%%writefile ")
                else source
            )
        implementation = "".join(source.split("\n", 1)[1] for source in code[1:-1])
        module = path.with_name("study_mitra.py").read_text(encoding="utf-8")
        self.assertEqual(implementation, module)
        self.assertNotIn("from study_mitra import", code[-1])

    def test_code_cells_write_separate_parts_without_importing_kernel_libraries(self):
        notebook = build_notebook()
        parts = [
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        ][1:-1]
        paths = [source.splitlines()[0] for source in parts]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(
            all(
                p.startswith("%%writefile /content/studymitra-runtime/cells/")
                for p in paths
            )
        )
        setup = "".join(notebook["cells"][2]["source"])
        self.assertIn("ensure_runtime(PROJECT_ROOT, LOCKED_REQUIREMENTS)", setup)
        self.assertNotIn("sys.version_info[:2] <= (3, 12)", setup)
        self.assertNotIn("Restart the runtime", setup)


if __name__ == "__main__":
    unittest.main()
