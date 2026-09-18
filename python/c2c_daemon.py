#!/usr/bin/env python3
"""
Cache-to-Cache (C2C) daemon.

A small loopback HTTP service that owns the `rosetta` runtime from
https://github.com/thu-nics/C2C (Cache-to-Cache: Direct Semantic Communication
Between Large Language Models). It loads a receiver (base) model, one or more
sharer (teacher) models and the trained C2C projector checkpoints, then answers
generation requests by fusing the sharers' KV-caches into the receiver instead
of exchanging text.

Why a daemon instead of one process per call: loading Qwen3-0.6B +
Qwen2.5-0.5B + 28 projectors takes tens of seconds, so a per-call process would
dominate the wall clock. The daemon keeps weights warm, unloads them after an
idle period, and serializes device access behind one lock (the underlying
models are not safe for concurrent generation).

The daemon is deliberately dependency-light at the HTTP layer (stdlib only) so
that a broken/absent extra package cannot take down the control plane.

Usage:
    python c2c_daemon.py --port 8765
    python c2c_daemon.py --print-config
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("c2c.daemon")

# Checkpoint layout: one projector definition per receiver layer.
PROJECTOR_RE = re.compile(r"projector_(\d+)\.json")

# --------------------------------------------------------------------------- #
# Built-in model registry
#
# Each entry names a published C2C Fuser checkpoint on the Hugging Face hub and
# the receiver/sharer pair it was trained for. `repo_subdir` is the directory
# inside the `nics-efc/C2C_Fuser` repository that must be downloaded.
# --------------------------------------------------------------------------- #
FUSER_REPO = "nics-efc/C2C_Fuser"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3_0.6b+qwen2.5_0.5b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "repo_subdir": "qwen3_0.6b+qwen2.5_0.5b_Fuser",
        "params": "0.6B receiver + 0.5B sharer (smallest published pair)",
    },
    "qwen3_0.6b+llama3.2_1b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "meta-llama/Llama-3.2-1B-Instruct",
        "repo_subdir": "qwen3_0.6b+llam3.2_1b_Fuser",
        "params": "0.6B receiver + 1B sharer (gated sharer repo)",
    },
    "qwen3_0.6b+qwen2.5_1.5b_math": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen2.5-Math-1.5B",
        "repo_subdir": "qwen3_0.6b+qwen2.5_1.5b_math_Fuser",
        "params": "0.6B receiver + 1.5B math specialist sharer",
    },
    "qwen3_0.6b+qwen3_4b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen3-4B",
        "repo_subdir": "qwen3_0.6b+qwen3_4b_Fuser",
        "params": "0.6B receiver + 4B sharer",
    },
    "qwen3_1.7b+qwen2.5_1.5b": {
        "base_model": "Qwen/Qwen3-1.7B",
        "teacher_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "repo_subdir": "qwen3_1.7b+qwen2.5_1.5b_Fuser",
        "params": "1.7B receiver + 1.5B sharer",
    },
    "qwen3_8b+qwen2.5_7b": {
        "base_model": "Qwen/Qwen3-8B",
        "teacher_model": "Qwen/Qwen2.5-7B-Instruct",
        "repo_subdir": "qwen3_8b+qwen2.5_7b_Fuser",
        "params": "8B receiver + 7B sharer (needs ~32GB+ unified memory)",
    },
}

DEFAULT_PAIR = "qwen3_0.6b+qwen2.5_0.5b"


# --------------------------------------------------------------------------- #
# Device and dtype policy
# --------------------------------------------------------------------------- #
def configure_hf_cache(repo_root: Optional[str]) -> Optional[str]:
    """
    Keep Hugging Face downloads inside the reproduction checkout.

    A sandboxed or read-only home directory makes the default
    ``~/.cache/huggingface`` unusable, and the weights belong next to the code
    that consumes them anyway. An explicit ``HF_HOME`` always wins.
    """
    existing = os.environ.get("HF_HOME")
    if existing:
        return existing
    if not repo_root:
        return None
    home = Path(repo_root).expanduser() / "models" / "hf"
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    os.environ["HF_HOME"] = str(home)
    return str(home)


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string against what this torch build actually offers."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # Let any op without an MPS kernel fall back to CPU instead of raising
        # NotImplementedError mid-generation. Slower for the offending op, but a
        # partially fused answer beats a crashed call.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str):
    """
    Pick a compute dtype.

    Apple's MPS backend historically lacked bfloat16 support; recent torch
    builds handle it, but float16 remains the faster and better-tested choice
    there. CPU stays float32 because half precision is poorly supported.
    """
    import torch

    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if requested and requested != "auto":
        return table[requested]
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
@dataclass
class LoadRequest:
    """One model-load request, resolved against the registry and the filesystem."""

    pair: str = DEFAULT_PAIR
    base_model: Optional[str] = None
    teacher_model: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    include_response: bool = False
    repo_root: Optional[str] = None


class C2CRuntime:
    """Owns the loaded RosettaModel and serializes all device work."""

    def __init__(self, repo_root: Optional[str] = None, idle_unload_seconds: int = 900):
        self.repo_root = Path(repo_root).expanduser() if repo_root else None
        self.idle_unload_seconds = int(idle_unload_seconds)
        self.lock = threading.RLock()
        self.loaded: Optional[Dict[str, Any]] = None
        self.last_used = 0.0
        self._reaper: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- setup helpers ----------------------------------------------------- #
    def _ensure_import_path(self) -> None:
        """Make the reproduced `rosetta` package importable."""
        if self.repo_root is None:
            return
        candidate = str(self.repo_root)
        if candidate not in sys.path and (self.repo_root / "rosetta").is_dir():
            sys.path.insert(0, candidate)
            LOG.info("added %s to sys.path for the rosetta package", candidate)

    def resolve_fuser_dir(self, pair: str, explicit: Optional[str]) -> str:
        """Find the directory holding projector_*.json / projector_*.pt."""
        if explicit:
            path = Path(explicit).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(f"checkpoints_dir does not exist: {path}")
            return str(path)

        entry = MODEL_REGISTRY.get(pair)
        if entry is None:
            raise KeyError(
                f"unknown pair '{pair}'; known pairs: {', '.join(sorted(MODEL_REGISTRY))}"
            )

        # A local download laid out as <root>/models/C2C_Fuser/<subdir>/final
        if self.repo_root is not None:
            for base in (self.repo_root / "models" / "C2C_Fuser", self.repo_root / "models"):
                cand = base / entry["repo_subdir"] / "final"
                if (cand / "projector_config.json").exists():
                    LOG.info("using local fuser checkpoint %s", cand)
                    return str(cand)

        # Fall back to the Hugging Face cache, downloading on first use.
        from huggingface_hub import snapshot_download

        LOG.info("downloading %s (%s) from the Hugging Face hub", pair, FUSER_REPO)
        root = snapshot_download(
            repo_id=FUSER_REPO, allow_patterns=[f"{entry['repo_subdir']}/*"]
        )
        cand = Path(root) / entry["repo_subdir"] / "final"
        if not (cand / "projector_config.json").exists():
            raise FileNotFoundError(
                f"downloaded {FUSER_REPO} but {cand}/projector_config.json is missing"
            )
        return str(cand)

    # -- load / unload ----------------------------------------------------- #
    def load(self, req: LoadRequest) -> Dict[str, Any]:
        import torch

        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self.lock:
            self._ensure_import_path()
            from rosetta.model.projector import load_projector
            from rosetta.model.wrapper import RosettaModel
            from rosetta.utils.evaluate import set_default_chat_template

            entry = MODEL_REGISTRY.get(req.pair, {})
            base_model = req.base_model or entry.get("base_model")
            teacher_model = req.teacher_model or entry.get("teacher_model")
            if not base_model or not teacher_model:
                raise ValueError(
                    "base_model and teacher_model are required when pair is not in the registry"
                )

            checkpoints_dir = self.resolve_fuser_dir(req.pair, req.checkpoints_dir)
            device = resolve_device(req.device)
            dtype = resolve_dtype(req.dtype, device)

            started = time.time()
            LOG.info(
                "loading receiver=%s sharer=%s device=%s dtype=%s",
                base_model,
                teacher_model,
                device,
                dtype,
            )

            tokenizer = AutoTokenizer.from_pretrained(str(base_model))
            set_default_chat_template(tokenizer, base_model)

            def _load_causal(path: str):
                kwargs: Dict[str, Any] = {"device_map": {"": device}}
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        str(path), dtype=dtype, **kwargs
                    ).eval()
                except TypeError:
                    # transformers < 4.56 spells this kwarg `torch_dtype`.
                    return AutoModelForCausalLM.from_pretrained(
                        str(path), torch_dtype=dtype, **kwargs
                    ).eval()

            receiver = _load_causal(base_model)
            sharer = _load_causal(teacher_model)

            # The checkpoint stores one projector per receiver layer plus a JSON
            # describing its architecture; weights load into the built module.
            ckpt = Path(checkpoints_dir)
            indices = sorted(
                int(match.group(1))
                for path in ckpt.glob("projector_*.json")
                if (match := PROJECTOR_RE.fullmatch(path.name))
            )
            if not indices:
                raise FileNotFoundError(f"no projector_<N>.json found in {checkpoints_dir}")

            projector_list = []
            for index in indices:
                cfg_path = ckpt / f"projector_{index}.json"
                projector = load_projector(str(cfg_path)).to(device)
                pt_path = ckpt / f"projector_{index}.pt"
                if pt_path.exists():
                    state = torch.load(pt_path, map_location=device, weights_only=True)
                    projector.load_state_dict(state, strict=False)
                projector.eval()
                projector_list.append(projector)

            if not projector_list:
                raise FileNotFoundError(f"no projector_*.json found in {checkpoints_dir}")

            model = RosettaModel(
                model_list=[receiver, sharer],
                base_model_idx=0,
                projector_list=projector_list,
                include_response=bool(req.include_response),
            ).to(device).eval()

            config_path = ckpt / "projector_config.json"
            if config_path.exists():
                model.load_projector_config(str(config_path))
            else:
                raise FileNotFoundError(f"{config_path} is missing; cannot map projectors")

            self.loaded = {
                "model": model,
                "tokenizer": tokenizer,
                "pair": req.pair,
                "base_model": base_model,
                "teacher_model": teacher_model,
                "checkpoints_dir": checkpoints_dir,
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "num_projectors": len(projector_list),
                "include_response": bool(req.include_response),
                "loaded_at": time.time(),
                "load_seconds": round(time.time() - started, 2),
            }
            self.last_used = time.time()
            LOG.info(
                "loaded %s in %.1fs (%d projectors)",
                req.pair,
                self.loaded["load_seconds"],
                len(projector_list),
            )
            return self.status()

    def unload(self) -> Dict[str, Any]:
        with self.lock:
            if self.loaded is not None:
                LOG.info("unloading %s", self.loaded.get("pair"))
                self.loaded = None
                self._release_memory()
            return self.status()

    @staticmethod
    def _release_memory() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:  # cleanup must never mask the real result
            LOG.debug("memory release failed", exc_info=True)

    # -- generation -------------------------------------------------------- #
    def generate(self, req: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        with self.lock:
            if self.loaded is None:
                raise RuntimeError("no model loaded; call the load action first")

            state = self.loaded
            model = state["model"]
            tokenizer = state["tokenizer"]
            device = state["device"]
            mode = req.get("mode", "c2c")

            messages = req.get("messages")
            if messages is None:
                prompt = req.get("prompt")
                if prompt is None:
                    raise ValueError("either prompt or messages is required")
                messages = [{"role": "user", "content": prompt}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(req.get("enable_thinking", False)),
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)
            prompt_tokens = int(inputs["input_ids"].shape[1])

            # kv_cache_index[0] governs the prompt (prefill) pass and
            # kv_cache_index[1] the decoded response. Value 1 selects sharer 1,
            # -1 means "no projection" (receiver-only baseline).
            #
            # Each entry is a (batch, section_len, 2) tensor: the wrapper derives
            # section boundaries from `.shape[1]` and reads only element [0,0,0]
            # as the sharer bitmask, so the leading section must span the prompt
            # minus its final token (the wrapper appends that one itself).
            sharer_mask = 1 if mode == "c2c" else -1
            instruction_index = (
                torch.tensor([sharer_mask, 0], dtype=torch.long)
                .repeat(max(prompt_tokens - 1, 1), 1)
                .unsqueeze(0)
                .to(device)
            )
            label_index = torch.tensor([[-1, 0]], dtype=torch.long).unsqueeze(0).to(device)

            generation: Dict[str, Any] = {
                "do_sample": bool(req.get("do_sample", False)),
                "max_new_tokens": int(req.get("max_new_tokens", 256)),
            }
            if generation["do_sample"]:
                generation["temperature"] = float(req.get("temperature", 1.0))
                generation["top_p"] = float(req.get("top_p", 1.0))
            for key in ("repetition_penalty",):
                if req.get(key) is not None:
                    generation[key] = float(req[key])

            started = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    kv_cache_index=[instruction_index, label_index],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation,
                )
            elapsed = time.time() - started
            self.last_used = time.time()

            generated = outputs[0, prompt_tokens:]
            new_tokens = int(generated.shape[0])
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            return {
                "text": answer,
                "mode": mode,
                "pair": state["pair"],
                "device": device,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": new_tokens,
                "seconds": round(elapsed, 3),
                "tokens_per_second": round(new_tokens / elapsed, 3) if elapsed > 0 else None,
            }

    # -- control ----------------------------------------------------------- #
    def status(self) -> Dict[str, Any]:
        with self.lock:
            info: Dict[str, Any] = {
                "loaded": self.loaded is not None,
                "idle_unload_seconds": self.idle_unload_seconds,
            }
            if self.loaded is not None:
                info.update(
                    {
                        "pair": self.loaded["pair"],
                        "base_model": self.loaded["base_model"],
                        "teacher_model": self.loaded["teacher_model"],
                        "checkpoints_dir": self.loaded["checkpoints_dir"],
                        "device": self.loaded["device"],
                        "dtype": self.loaded["dtype"],
                        "num_projectors": self.loaded["num_projectors"],
                        "include_response": self.loaded["include_response"],
                        "load_seconds": self.loaded["load_seconds"],
                        "idle_seconds": round(time.time() - self.last_used, 1),
                    }
                )
            return info

    def start_reaper(self) -> None:
        if self.idle_unload_seconds <= 0:
            return

        def loop() -> None:
            while not self._stop.wait(30):
                with self.lock:
                    idle = self.loaded is not None and (
                        time.time() - self.last_used > self.idle_unload_seconds
                    )
                if idle:
                    LOG.info("idle timeout reached; releasing weights")
                    self.unload()

        self._reaper = threading.Thread(target=loop, name="c2c-reaper", daemon=True)
        self._reaper.start()

    def shutdown(self) -> None:
        self._stop.set()
        self.unload()


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
def build_handler(runtime: C2CRuntime):
    """Build the request handler bound to one runtime instance."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "c2c-daemon/0.1"
        protocol_version = "HTTP/1.1"

        # -- plumbing ------------------------------------------------------ #
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
            LOG.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON body: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        # -- routes -------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                if path in ("/", "/health"):
                    self._send(200, {"ok": True, "service": "c2c-daemon", "status": runtime.status()})
                elif path == "/status":
                    self._send(200, runtime.status())
                elif path == "/models":
                    self._send(
                        200,
                        {
                            "fuser_repo": FUSER_REPO,
                            "default_pair": DEFAULT_PAIR,
                            "pairs": MODEL_REGISTRY,
                        },
                    )
                else:
                    self._send(404, {"ok": False, "error": f"unknown path {path}"})
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                body = self._read_json()
                if path == "/load":
                    self._send(200, {"ok": True, "status": runtime.load(LoadRequest(**body))})
                elif path == "/generate":
                    self._send(200, {"ok": True, **runtime.generate(body)})
                elif path == "/unload":
                    self._send(200, {"ok": True, "status": runtime.unload()})
                else:
                    self._send(404, {"ok": False, "error": f"unknown path {path}"})
            except (TypeError, ValueError) as exc:
                self._send(400, {"ok": False, "error": f"bad request: {exc}"})
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)

        def _fail(self, exc: Exception) -> None:
            LOG.error("request failed: %s", exc)
            LOG.debug("%s", traceback.format_exc())
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return Handler


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C2C (Cache-to-Cache) daemon")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    parser.add_argument("--port", type=int, default=8765, help="TCP port")
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("C2C_REPO_ROOT", ""),
        help="root of the reproduced thu-nics/C2C checkout (for `rosetta` and local checkpoints)",
    )
    parser.add_argument(
        "--idle-unload",
        type=int,
        default=int(os.environ.get("C2C_IDLE_UNLOAD", "900")),
        help="seconds of inactivity before weights are released (0 disables)",
    )
    parser.add_argument(
        "--preload",
        default=os.environ.get("C2C_PRELOAD", ""),
        help="pair name to load at startup (empty means lazy loading)",
    )
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, or float32")
    parser.add_argument("--print-config", action="store_true", help="print the resolved config as JSON")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    repo_root = args.repo_root or None
    hf_home = configure_hf_cache(repo_root)
    runtime = C2CRuntime(repo_root=repo_root, idle_unload_seconds=args.idle_unload)

    resolved = {
        "host": args.host,
        "port": args.port,
        "repo_root": repo_root,
        "hf_home": hf_home,
        "idle_unload": args.idle_unload,
        "preload": args.preload or None,
    }
    if args.print_config:
        print(json.dumps(resolved, indent=2))
        return 0

    runtime._ensure_import_path()  # noqa: SLF001 - deliberate early path check

    if args.preload:
        try:
            runtime.load(
                LoadRequest(pair=args.preload, device=args.device, dtype=args.dtype, repo_root=repo_root)
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error("preload failed: %s", exc)
            LOG.debug("%s", traceback.format_exc())

    runtime.start_reaper()
    server = ThreadingHTTPServer((args.host, args.port), build_handler(runtime))
    server.daemon_threads = True
    LOG.info("c2c daemon listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutting down")
    finally:
        server.shutdown()
        runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
