#!/usr/bin/env python3
"""
Bootstrap the Cache-to-Cache runtime that dsh-plugin-c2c drives.

The plugin ships the transport (a Node tool surface plus a local daemon) but not
the model weights or the PyTorch environment on top of PyTorch. This script sets
that up: it clones https://github.com/thu-nics/C2C, creates a virtual
environment with the right pinned dependencies, and verifies the result.

Model weights are deliberately NOT downloaded here. They are ~3.2 GB and only one
pair is needed at a time, so the daemon fetches the selected pair on first load
and caches it under <root>/models/hf.

Usage:
    python dsh_c2c_setup.py --root ~/factor_digging/c2c
    python dsh_c2c_setup.py --root ./c2c --pair qwen3_0.6b+qwen2.5_0.5b --verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from typing import List, Optional, Sequence

UPSTREAM_REPO = "https://github.com/thu-nics/C2C.git"

# transformers is an UPPER bound here, not a floor. rosetta clones KV caches by
# appending to `DynamicCache.key_cache`, which is a plain list only through
# 4.54.x. 4.55 wraps it in an object without .append(); 4.56 removes the
# attribute. Pinning avoids a failure that only shows up at model-load time.
TRANSFORMERS_PIN = "transformers==4.52.4"
BASE_REQUIREMENTS: Sequence[str] = (
    "torch",
    TRANSFORMERS_PIN,
    "accelerate",  # required by transformers whenever device_map= is passed
    "datasets",  # imported at module level by rosetta.train.dataset_adapters
    "sentencepiece",
    "protobuf",
    "huggingface_hub",
)

# The interpreter used for the venv. C2C declares requires-python >=3.10.
MIN_PY = (3, 10)
MAX_PY = (3, 13)  # torch wheels for 3.14 were not available at time of writing


def log(message: str) -> None:
    print(f"[dsh-c2c] {message}", flush=True)


def run(cmd: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, streaming nothing, and raise on failure with its output."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        tail = (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{tail}")
    return result


def find_python() -> str:
    """
    Find an interpreter torch publishes wheels for.

    A too-new interpreter is the single most common way this bootstrap fails, so
    it is detected up front rather than surfacing as a pip resolution error deep
    into a multi-gigabyte install.
    """
    override = os.environ.get("DSH_C2C_PYTHON")
    if override:
        return override

    candidates: List[str] = []
    for minor in range(MAX_PY[1], MIN_PY[1] - 1, -1):
        candidates += [f"python3.{minor}", f"/opt/homebrew/bin/python3.{minor}", f"/usr/local/bin/python3.{minor}"]
    candidates += [sys.executable, "python3", "python"]

    seen = set()
    for candidate in candidates:
        resolved = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if not resolved or resolved in seen or not Path(resolved).exists():
            continue
        seen.add(resolved)
        try:
            probe = run([resolved, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
        except Exception:
            continue
        version = tuple(int(p) for p in probe.stdout.strip().split("."))
        if MIN_PY <= version <= MAX_PY:
            log(f"using interpreter {resolved} (Python {version[0]}.{version[1]})")
            return resolved
        log(f"skipping {resolved}: Python {version[0]}.{version[1]} is outside {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]}")
    raise SystemExit(
        f"no suitable Python found. C2C needs {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]} "
        f"(PyTorch wheel availability). Install one, e.g. `brew install python@3.12`, "
        f"or point DSH_C2C_PYTHON at an existing interpreter."
    )


def ensure_checkout(root: Path, repo_url: str, update: bool) -> None:
    """Clone the C2C checkout, or update it when it is already a git repo."""
    if (root / "rosetta").is_dir():
        log(f"checkout already present at {root}")
        if update and (root / ".git").is_dir():
            log("updating checkout (git pull --ff-only)")
            try:
                run(["git", "-C", str(root), "pull", "--ff-only"])
            except RuntimeError as exc:
                log(f"update skipped: {exc}")
        return

    root.parent.mkdir(parents=True, exist_ok=True)
    log(f"cloning {repo_url} -> {root}")
    run(["git", "clone", "--depth", "1", repo_url, str(root)])
    if not (root / "rosetta").is_dir():
        raise SystemExit(f"clone succeeded but {root}/rosetta is missing — unexpected upstream layout")


def venv_python(root: Path) -> Path:
    """Path to the venv interpreter, accounting for Windows layout."""
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def ensure_venv(root: Path, interpreter: str, recreate: bool) -> Path:
    """Create the virtual environment and install pinned requirements."""
    target = root / ".venv"
    python = venv_python(root)

    if recreate and target.exists():
        log("removing existing .venv (--recreate)")
        shutil.rmtree(target)

    if not python.exists():
        log(f"creating virtual environment at {target}")
        venv.EnvBuilder(with_pip=True, clear=False).create(str(target))
    else:
        log(f"reusing virtual environment at {target}")

    log("upgrading pip")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    log(f"installing runtime dependencies ({len(BASE_REQUIREMENTS)} packages; this pulls PyTorch)")
    run([str(python), "-m", "pip", "install", "--quiet", *BASE_REQUIREMENTS])
    return python


def verify(root: Path, python: Path) -> dict:
    """Import the runtime pieces and report what resolved."""
    script = f"""
import json, sys
sys.path.insert(0, {str(root)!r})
import torch, transformers
from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import load_projector
from rosetta.utils.core import sharers_to_mask
out = {{
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    "cuda": bool(torch.cuda.is_available()),
    "rosetta_import": True,
}}
print(json.dumps(out))
"""
    result = run([str(python), "-c", script])
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Set up the Cache-to-Cache runtime driven by dsh-plugin-c2c"
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("C2C_REPO_ROOT") or str(Path.home() / "factor_digging" / "c2c"),
        help="where to put the C2C checkout (default: ~/factor_digging/c2c)",
    )
    parser.add_argument("--repo-url", default=UPSTREAM_REPO, help="upstream repository to clone")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild .venv")
    parser.add_argument("--no-update", action="store_true", help="skip git pull on an existing checkout")
    parser.add_argument("--verify", action="store_true", help="report versions and device availability")
    parser.add_argument("--print-config", action="store_true", help="print the plugin config snippet")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    log(f"target root: {root}")

    interpreter = find_python()
    ensure_checkout(root, args.repo_url, update=not args.no_update)
    python = ensure_venv(root, interpreter, recreate=args.recreate)

    info = verify(root, python)
    log("runtime ready")
    print(json.dumps(info, indent=2))

    device = "mps" if info["mps"] else ("cuda" if info["cuda"] else "cpu")
    log(f"device will resolve to: {device}")
    log(f"model weights are downloaded on first use into {root / 'models' / 'hf'}")

    if args.print_config:
        print("\n# Add to your dsh profile's cordis.patch.yml:")
        print("- id: c2c")
        print("  config:")
        print(f"    repoRoot: '{root}'")
        print(f"    pythonPath: '{python}'")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)
    except RuntimeError as exc:
        log(f"FAILED: {exc}")
        raise SystemExit(1)
