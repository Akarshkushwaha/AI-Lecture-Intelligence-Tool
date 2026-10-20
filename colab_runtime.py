"""Isolated notebook runtime; standard library only, safe in a Colab kernel.

This source is embedded in Cell 1 so the notebook stays standalone.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

UV_VERSION = "0.11.11"
RUNTIME_SCHEMA = "studymitra-isolated-v1"
HEALTHCHECK = """import sys
import numpy, numpy.random, pandas, spacy, gradio, fastapi, pydantic, torch, ctranslate2
from faster_whisper import WhisperModel
from transformers import BartForConditionalGeneration, MarianMTModel, pipeline
from keybert import KeyBERT
import matplotlib.pyplot as plt
assert plt.get_backend().lower() == "agg", plt.get_backend()
figure = plt.figure()
figure.canvas.draw()
plt.close(figure)
assert sys.version_info[:2] == (3, 12), sys.version
assert numpy.__version__ == "1.26.4", numpy.__version__
assert pandas.__version__ == "2.2.3", pandas.__version__
assert fastapi.__version__ == "0.112.2", fastapi.__version__
assert pydantic.__version__ == "2.10.6", pydantic.__version__
assert torch.__version__.split("+")[0] == "2.5.1", torch.__version__
assert ctranslate2.__version__ == "4.6.0", ctranslate2.__version__
spacy.load("en_core_web_sm")
if torch.cuda.is_available():
    import ctypes
    ctypes.CDLL("libcublas.so.12")
    ctypes.CDLL("libcudnn.so.9")
    torch.zeros(1, device="cuda")
print("Runtime health check passed | Python", sys.version.split()[0],
      "| NumPy", numpy.__version__, "| pandas", pandas.__version__,
      "| FastAPI", fastapi.__version__, "| Device", "CUDA" if torch.cuda.is_available() else "CPU", flush=True)
"""


def run_setup(label, command, env=None):
    """Stream child output and stop the child when the notebook is interrupted."""
    print(f"\n--- {label} ---", flush=True)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    print(f"[OK] {label}", flush=True)


def runtime_python(root):
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return Path(root) / "venv" / directory / executable


def runtime_environment(root):
    root = Path(root)
    env = dict(os.environ)
    # Inherited Colab paths must not inject its installed Python packages.
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        HF_HOME=str(root / "models"),
        GRADIO_ANALYTICS_ENABLED="False",
        UV_CACHE_DIR=str(root / "cache" / "uv"),
        UV_PYTHON_INSTALL_DIR=str(root / "python"),
        # Colab exports a notebook-only backend. The app is a separate server.
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(root / "matplotlib"),
    )
    # CTranslate2 loads CUDA libraries dynamically; use this environment's copies.
    libraries = sorted(root.glob("venv/lib/python*/site-packages/nvidia/*/lib"))
    if libraries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(p) for p in libraries] + [env.get("LD_LIBRARY_PATH", "")]
        )
    return env


def ensure_uv(root, runner=run_setup):
    """Install only the standalone uv tool into a private bootstrap directory."""
    bootstrap = Path(root) / "bootstrap"
    uv = (
        bootstrap
        / ("Scripts" if os.name == "nt" else "bin")
        / ("uv.exe" if os.name == "nt" else "uv")
    )
    if not uv.exists():
        runner(
            "Install environment manager",
            [
                sys.executable,
                "-u",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(bootstrap),
                f"uv=={UV_VERSION}",
            ],
            env=runtime_environment(root),
        )
    if not uv.exists():
        raise RuntimeError(f"uv executable not found after installation: {uv}")
    return uv


def ensure_runtime(root, locked_requirements, runner=run_setup, uv=None):
    """Idempotent setup: never install scientific packages in the notebook kernel."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "cells").mkdir(exist_ok=True)
    env = runtime_environment(root)
    uv = Path(uv) if uv else ensure_uv(root, runner)
    python = runtime_python(root)
    lockfile = root / "requirements.lock"
    lockfile.write_text(locked_requirements, encoding="utf-8")
    fingerprint = hashlib.sha256(
        (RUNTIME_SCHEMA + locked_requirements).encode("utf-8")
    ).hexdigest()
    marker = root / "ready.json"
    try:
        ready = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ready = {}
    reusable = python.exists() and ready.get("fingerprint") == fingerprint
    if not python.exists():
        runner(
            "Create isolated Python 3.12 environment",
            [str(uv), "venv", "--python", "3.12", str(root / "venv")],
            env=env,
        )
    if not reusable:
        # An interrupted or changed setup must never retain a ready marker.
        marker.unlink(missing_ok=True)
        runner(
            "Install locked application dependencies (first run downloads packages)",
            [str(uv), "pip", "sync", "--python", str(python), str(lockfile)],
            env=env,
        )
    else:
        print("Reusing the verified application environment.", flush=True)
    try:
        runner(
            "Check installed dependency consistency",
            [str(uv), "pip", "check", "--python", str(python)],
            env=runtime_environment(root),
        )
        runner(
            "Check imports and binary compatibility",
            [str(python), "-I", "-u", "-c", HEALTHCHECK],
            env=runtime_environment(root),
        )
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    marker.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    return python


def assemble_source(root, cell_names):
    """Overwrite the assembled module; rerunning cells cannot duplicate definitions."""
    root = Path(root)
    paths = [root / "cells" / name for name in cell_names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Run the code cells in order first. Missing: {', '.join(missing)}"
        )
    source = "".join(p.read_text(encoding="utf-8") for p in paths)
    compile(source, "study_mitra.py", "exec")
    target = root / "study_mitra.py"
    target.write_text(source, encoding="utf-8")
    return target


def launch_notebook(root, cell_names, share=True, runner=run_setup):
    root = Path(root).resolve()
    if not (root / "ready.json").exists():
        raise RuntimeError("Run Cell 1 successfully before launching.")
    assemble_source(root, cell_names)
    server = root / "server.py"
    server.write_text(
        "from study_mitra import build_app\n"
        "demo = build_app()\n"
        "demo.queue(max_size=5, default_concurrency_limit=1).launch("
        f"share={share!r}, debug=True, show_error=True, inline=False, inbrowser=False)\n",
        encoding="utf-8",
    )
    # -I prevents notebook/site customizations from importing Colab's NumPy.
    command = [
        str(runtime_python(root)),
        "-I",
        "-u",
        "-c",
        f"import sys, runpy; sys.path.insert(0, {str(root)!r}); "
        f"runpy.run_path({str(server)!r}, run_name='__main__')",
    ]
    print(
        "Open the public Gradio URL below. Keep this cell running while using the app.",
        flush=True,
    )
    runner(
        "Start StudyMitra in its isolated environment",
        command,
        env=runtime_environment(root),
    )
