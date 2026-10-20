"""Run the standalone notebook in order without changing the host Python stack.

Usage: python tests/validate_colab.py --root .cache/colab-validation
The host needs IPython. The application gets its own Python 3.12 environment.
"""

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from IPython.core.interactiveshell import InteractiveShell

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from colab_runtime import launch_notebook, runtime_environment, runtime_python
from sync_notebook import build_notebook


def validate(root):
    import numpy as host_numpy

    original_numpy = host_numpy.__version__
    root = Path(root).resolve()
    notebook = build_notebook(root.as_posix())
    shell = InteractiveShell.instance()
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    print(f"HOST: Python {sys.version.split()[0]}, NumPy {original_numpy}", flush=True)
    for index, source in enumerate(code[:-1], 1):
        result = shell.run_cell(source)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Notebook code cell {index} failed") from (
                result.error_before_exec or result.error_in_exec
            )
    # Rerunning Cell 1 is part of the supported sequence, not a cell to skip.
    result = shell.run_cell(code[0])
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError("Rerunning setup failed")
    assert sys.modules["numpy"] is host_numpy
    assert host_numpy.__version__ == original_numpy
    print(
        "Host NumPy remained loaded and unchanged after setup, source cells, and setup rerun.",
        flush=True,
    )

    processes = []
    log_path = root / "validation-server.log"
    log = log_path.open("w", encoding="utf-8")

    def start_server(label, command, env):
        processes.append(
            subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        )

    # Execute the notebook's actual launch cell, changing only share=True for this local check.
    def local_launch(project_root, cell_names, share):
        launch_notebook(project_root, cell_names, share=False, runner=start_server)

    shell.user_ns["launch_notebook"] = local_launch
    try:
        result = shell.run_cell(code[-1])
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError("Notebook launch cell failed")
        deadline = time.monotonic() + 180
        url = None
        while time.monotonic() < deadline:
            if processes[0].poll() is not None:
                raise RuntimeError(log_path.read_text(encoding="utf-8"))
            match = re.search(
                r"http://127\.0\.0\.1:\d+", log_path.read_text(encoding="utf-8")
            )
            if match:
                url = match.group(0)
                break
            time.sleep(0.5)
        if not url:
            raise TimeoutError(
                "Server startup timed out: " + log_path.read_text(encoding="utf-8")
            )
        probe = """import sys
from gradio_client import Client
client = Client(sys.argv[1], verbose=False)
result = client.submit(None, "Auto", "VAD Enabled", "Auto-detect",
                       api_name="/run_transcribe").result(timeout=30)
assert "Please upload" in result[0], result
client.close()
print("Notebook server HTTP queue request passed.")
"""
        subprocess.run(
            [str(runtime_python(root)), "-I", "-u", "-c", probe, url],
            env=runtime_environment(root),
            check=True,
        )
    finally:
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()
    print(
        "PASS: every notebook code cell ran in order; setup rerun and queued request passed.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO / ".cache/colab-validation"))
    args = parser.parse_args()
    validate(args.root)
