"""Windows DLL bootstrap for TensorRT 10 + ONNX Runtime GPU + PyTorch.

Why this exists (all discovered by running, not guessing):

1. ORT-GPU 1.23.2's TensorrtExecutionProvider needs nvinfer_10.dll.
   ORT's embedded TRT EP strings confirm the supported baseline is
   TensorRT 10.13; the matching runtime comes from the pip wheel
   tensorrt-cu12==10.13.3.9.post1 (venv site-packages/tensorrt_libs).
   A TRT 10.0.1 DLL set in tensorrt_libs/ is kept as fallback.

2. ORT's CUDA 12 EP needs cuBLAS/cuDNN/cudart 12 DLLs from the nvidia-*
   pip wheels (handled by nvidia_ort_loader.py in this directory).

3. torch 2.7.1+cu118 must be imported FIRST so its cuDNN 9 DLL wins the
   process-wide name; the cu12 wheels' cudnn must stay unloaded.

Usage (call before creating any onnxruntime InferenceSession):
    import trt_bootstrap
    trt_bootstrap.setup()
"""
import ctypes
import os
import sys

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRT_LIBS = os.path.join(_PROJECT, "tensorrt_libs")

# TRT 10 main DLLs in dependency-safe order (ORT 1.23.2 links nvinfer_10.dll)
_TRT_DLLS = (
    "nvinfer_10.dll",
    "nvinfer_lean_10.dll",
    "nvinfer_dispatch_10.dll",
    "nvinfer_plugin_10.dll",
    "nvinfer_vc_plugin_10.dll",
    "nvonnxparser_10.dll",
)


def setup(verbose: bool = True) -> list:
    """Import torch, preload cu12 DLLs, then preload TensorRT DLLs.

    Returns the list of TRT DLLs successfully loaded.
    """
    # 1) torch first: its cuDNN 9 (CUDA 11.8 build) must own the name
    import torch  # noqa: F401

    # 2) preload CUDA 12 runtime DLLs from the nvidia pip wheels
    import nvidia_ort_loader

    nvidia_ort_loader.setup()

    # 3) use exactly ONE TRT source dir to avoid version mixing:
    #    prefer pip-wheel TRT 10.13 DLLs (matches ORT 1.23.2); fall back to
    #    the older 10.0.1 DLL set shipped in tensorrt_libs/. Mixing a 10.13
    #    nvinfer_10.dll with a 10.0.1 nvinfer_builder_resource_10.dll fails
    #    engine builds with "Assertion validateCaskKLSize failed" (observed).
    _wheel = os.path.join(
        _PROJECT, "venv", "Lib", "site-packages", "tensorrt_libs"
    )
    if os.path.isfile(os.path.join(_wheel, "nvinfer_10.dll")):
        lib_dir = _wheel
    elif os.path.isdir(_TRT_LIBS):
        lib_dir = _TRT_LIBS
    else:
        lib_dir = ""

    loaded = []
    if lib_dir:
        os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(lib_dir)
        except (OSError, AttributeError):
            pass
        for dll in _TRT_DLLS:
            full = os.path.join(lib_dir, dll)
            if not os.path.isfile(full):
                continue
            try:
                ctypes.CDLL(full)
                loaded.append(dll)
            except OSError as exc:
                if verbose:
                    print(f"[trt_bootstrap] FAILED {dll}: {exc}", file=sys.stderr)
    if verbose:
        print(f"[trt_bootstrap] TRT dir: {lib_dir}")
        print(f"[trt_bootstrap] loaded {len(loaded)} TRT DLLs: {loaded}")
    return loaded


if __name__ == "__main__":
    setup()
