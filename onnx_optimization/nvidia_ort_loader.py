"""Windows DLL loader so ONNX Runtime GPU (CUDA 12 EP) and PyTorch (CUDA 11.8)
can work in the SAME process.

Constraints discovered by testing on this project:

1. ORT 1.23.2's onnxruntime_providers_cuda.dll needs CUDA 12.x DLLs
   (cublasLt64_12, cudart64_12, cufft64_11). They are bundled in the
   nvidia-* pip wheels and are not on PATH, so ORT silently falls back
   to CPU unless they are preloaded by absolute path.
2. torch 2.7.1+cu118 ships its own cudnn64_9.dll. If the cu12 wheel
   version is loaded first, torch's load fails with WinError 127
   (procedure not found). Importing torch FIRST claims the cuDNN name,
   after which the remaining cu12 DLLs (cublas/cudart/cufft/nvrtc)
   can be preloaded for ORT.

Location of the nvidia wheels is resolved from, in order: the running
interpreter's site-packages, the repository's venv, then the directory
holding this file.

Usage (call before creating any onnxruntime InferenceSession):

    import nvidia_ort_loader
    nvidia_ort_loader.setup()
"""
import ctypes
import os
import sysconfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)


def _find_nvidia_dir():
    """Return the site-packages that contains the nvidia/* wheel DLLs."""
    candidates = []
    try:
        candidates.append(sysconfig.get_paths()["purelib"])
    except (KeyError, OSError):
        pass
    candidates.append(os.path.join(_PROJECT, "venv", "Lib", "site-packages"))
    candidates.append(_HERE)
    for parent in candidates:
        if os.path.isdir(os.path.join(parent, "nvidia")):
            return os.path.join(parent, "nvidia")
    return ""


# cu12 dirs that can be preloaded after torch; "cudnn" is excluded because
# torch's cu11 cuDNN 9 must win the name.
_PRELOAD = ("cublas", "cuda_runtime", "cuda_nvrtc", "cufft")


def setup() -> int:
    """Import torch (if available), then preload non-cudnn cu12 DLLs.

    Returns the number of NVIDIA DLLs successfully preloaded.
    """
    try:
        import torch  # noqa: F401  must load its cu11 cuDNN first
    except Exception:
        pass
    loaded = 0
    nv_dir = _find_nvidia_dir()
    if not nv_dir:
        return 0
    for sub in _PRELOAD:
        b = os.path.join(nv_dir, sub, "bin")
        if not os.path.isdir(b):
            continue
        for dll in sorted(os.listdir(b)):
            if dll.lower().endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(b, dll))
                    loaded += 1
                except OSError:
                    pass
    return loaded


if __name__ == "__main__":
    print(f"preloaded {setup()} NVIDIA cu12 DLLs (cudnn skipped for torch)")
