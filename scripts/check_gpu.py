"""Verify that the installed PyTorch build can actually run on the local GPU (e.g. RTX 5070, Blackwell sm_120).

Usage:  python scripts/check_gpu.py
Blackwell consumer GPUs (compute capability 12.0) need a PyTorch build compiled for sm_120, i.e. a CUDA 12.8+
wheel (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu128`). This script does not
assume anything; it reports what it finds and runs a tiny matmul + BF16 autocast check.
"""
import json
import sys

import torch


def main():
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        info["verdict"] = "NO CUDA DEVICE VISIBLE - training would run on CPU"
        print(json.dumps(info, indent=2))
        return 1
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(dev)
    arch = f"sm_{cap[0]}{cap[1]}"
    info.update({
        "device": torch.cuda.get_device_name(dev), "capability": cap, "arch": arch,
        "arch_list": torch.cuda.get_arch_list(),
        "total_vram_gb": round(torch.cuda.get_device_properties(dev).total_memory / 2**30, 2),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    })
    info["arch_in_build"] = arch in info["arch_list"] or any(a.startswith("compute_") and int(a.split("_")[1]) <= cap[0] * 10 + cap[1] for a in info["arch_list"])
    try:
        a = torch.randn(1024, 1024, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16 if info["bf16_supported"] else torch.float16):
            b = (a @ a).float().sum().item()
        torch.cuda.synchronize()
        info["kernel_test"] = "ok" if b == b else "nan"
    except Exception as e:  # e.g. "no kernel image is available for execution on the device"
        info["kernel_test"] = f"FAILED: {e}"
    info["verdict"] = "OK" if info["kernel_test"] == "ok" else "INSTALL A PYTORCH BUILD THAT SUPPORTS THIS GPU"
    print(json.dumps(info, indent=2, default=str))
    return 0 if info["verdict"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
