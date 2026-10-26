"""Methods 1-5: device, mixed precision, fast attention, torch.compile, multi-GPU DDP.

Methods 1 (single GPU) and 5 (DDP) are approval-needed: the code paths exist but refuse to
select CUDA or build a process group unless the matching *_approved flag is True. Nothing here
rents hardware, and no process group is created in tests (that would open sockets).
"""
import contextlib, shlex, time
from typing import Callable, Optional

import torch


class ApprovalRequired(RuntimeError):
    pass


# ---- 1 single GPU -----------------------------------------------------------------------------
def resolve_device(device: str = "cpu", gpu_approved: bool = False) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if not gpu_approved:
        raise ApprovalRequired("GPU use needs Latoya's OK (price cap + time limit) -> set gpu_approved=True")
    if not torch.cuda.is_available():
        if device == "auto":
            return torch.device("cpu")
        raise RuntimeError("cuda requested but no CUDA device is visible")
    return torch.device("cuda")


# ---- 2 mixed precision ------------------------------------------------------------------------
def autocast(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---- 3 fast attention -------------------------------------------------------------------------
_BACKENDS = {"flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}


def attention_ctx(kind: str):
    """The model already calls F.scaled_dot_product_attention; this pins its kernel.
    'flash' uses FlashAttention when the build/device supports it (CPU builds ship a flash kernel)."""
    if kind in ("default", "auto", "", None):
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel(getattr(SDPBackend, _BACKENDS[kind]))


def flash_available(device: torch.device = torch.device("cpu")) -> bool:
    try:
        q = torch.randn(1, 2, 8, 16, device=device)
        with attention_ctx("flash"):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        return True
    except RuntimeError:
        return False


# ---- 4 torch.compile --------------------------------------------------------------------------
def maybe_compile(model, enabled: bool, probe: Optional[Callable] = None):
    """Returns (callable_model, status). Falls back to eager if inductor can't build (e.g. no Python.h)."""
    if not enabled:
        return model, "off"
    import warnings
    try:
        warnings.filterwarnings("ignore", message="Can't find Python.h")
        cm = torch.compile(model)
        if probe is not None:
            probe(cm)          # forces the compile now so failures surface here, not mid-run
        return cm, "compiled"
    except Exception as e:     # noqa: BLE001 - any backend failure means eager
        torch._dynamo.reset()
        return model, f"fallback-eager ({type(e).__name__}: {str(e).splitlines()[0][:120]})"


def time_steps(step_fn: Callable, n: int = 3, warm: int = 1) -> float:
    for _ in range(warm):
        step_fn()
    t = time.perf_counter()
    for _ in range(n):
        step_fn()
    return (time.perf_counter() - t) / n


def autotune(make_step: Callable[[str, str], Callable], precisions=("fp32", "bf16"),
             attentions=("default", "math", "flash"), n: int = 2) -> dict:
    """Time a few real train steps per (precision, attention) and keep the fastest. The probe
    steps are real updates on a throwaway copy, and their wall time counts against the budget."""
    res = {}
    for p in precisions:
        for a in attentions:
            try:
                res[(p, a)] = time_steps(make_step(p, a), n=n)
            except RuntimeError:
                pass
    best = min(res, key=res.get)
    return {"precision": best[0], "attention": best[1],
            "timings_s": {f"{p}/{a}": round(v, 4) for (p, a), v in res.items()}}


# ---- 5 DDP (torchrun) -------------------------------------------------------------------------
def torchrun_command(nproc: int, args: list) -> str:
    return " ".join(["torchrun", "--standalone", f"--nproc_per_node={int(nproc)}", "-m",
                     "artemis_dense.accel.trainer"] + [shlex.quote(a) for a in args])


def setup_ddp(model, approved: bool = False):
    """Wrap in DistributedDataParallel. Needs approval (multi-GPU rental) and a torchrun env."""
    if not approved:
        raise ApprovalRequired("multi-GPU DDP needs Latoya's OK (per-GPU price cap + time limit)")
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    if "RANK" not in os.environ:
        raise RuntimeError("launch with torchrun (see torchrun_command)")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        model = model.cuda(local)
        return DDP(model, device_ids=[local])
    return DDP(model)
