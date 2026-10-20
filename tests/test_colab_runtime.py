"""Verify isolation and rerun behavior without installing packages in the kernel."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from colab_runtime import (
    assemble_source,
    ensure_runtime,
    launch_notebook,
    runtime_environment,
    runtime_python,
)


class RuntimeTests(unittest.TestCase):
    def test_repeated_setup_reuses_environment_and_keeps_kernel_packages_untouched(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def runner(label, command, env=None):
                calls.append(command)
                if "venv" in command:
                    executable = runtime_python(root)
                    executable.parent.mkdir(parents=True)
                    executable.touch()

            with patch.dict(os.environ, {"PYTHONPATH": "/host/scientific-packages"}):
                ensure_runtime(root, "numpy==1.26.4\n", uv="uv", runner=runner)
                ensure_runtime(root, "numpy==1.26.4\n", uv="uv", runner=runner)
            self.assertEqual(sum("venv" in c for c in calls), 1)
            self.assertEqual(sum("sync" in c for c in calls), 1)
            self.assertEqual(sum("-I" in c for c in calls), 2)
            sync = next(c for c in calls if "sync" in c)
            self.assertEqual(
                sync[sync.index("--python") + 1], str(runtime_python(root))
            )
            self.assertTrue((root / "ready.json").exists())

    def test_failed_or_changed_setup_cannot_leave_a_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = runtime_python(root)
            python.parent.mkdir(parents=True)
            python.touch()
            (root / "ready.json").write_text('{"fingerprint":"old"}')

            def fail(label, command, env=None):
                if "-c" in command:
                    raise subprocess.CalledProcessError(1, command)

            with self.assertRaises(subprocess.CalledProcessError):
                ensure_runtime(root, "different-lock", uv="uv", runner=fail)
            self.assertFalse((root / "ready.json").exists())

    def test_child_environment_does_not_inherit_colab_python_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
            library.mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "PYTHONPATH": "host",
                    "PYTHONHOME": "host",
                    "VIRTUAL_ENV": "host",
                    "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                    "MPLCONFIGDIR": "/host/matplotlib",
                },
            ):
                env = runtime_environment(root)
            for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
                self.assertNotIn(key, env)
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(env["MPLBACKEND"], "Agg")
            self.assertEqual(env["MPLCONFIGDIR"], str(root / "matplotlib"))
            self.assertIn(str(library), env["LD_LIBRARY_PATH"])

    def test_rerunning_source_cell_overwrites_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cells").mkdir()
            first = root / "cells/02.py"
            first.write_text("value = 1\n")
            (root / "cells/03.py").write_text("result = value + 1\n")
            assemble_source(root, ["02.py", "03.py"])
            first.write_text("value = 2\n")
            assembled = assemble_source(root, ["02.py", "03.py"]).read_text()
            self.assertEqual(assembled, "value = 2\nresult = value + 1\n")
            with self.assertRaises(RuntimeError):
                assemble_source(root, ["missing.py"])

    def test_launch_uses_verified_child_python_and_streams_server_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cells").mkdir()
            (root / "cells/02.py").write_text("def build_app(): pass\n")
            with self.assertRaises(RuntimeError):
                launch_notebook(root, ["02.py"])
            (root / "ready.json").write_text("{}")
            calls = []
            launch_notebook(
                root,
                ["02.py"],
                share=False,
                runner=lambda label, command, env: calls.append(command),
            )
            self.assertEqual(calls[0][0], str(runtime_python(root)))
            self.assertIn("-I", calls[0])
            self.assertIn("share=False", (root / "server.py").read_text())
