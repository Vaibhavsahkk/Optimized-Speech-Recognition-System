"""Quick empirical check: does TensorrtExecutionProvider become ACTIVE
after trt_bootstrap.setup()? Creates a session but does NOT run inference
(engine build happens on first run, which is a separate longer test).

Uses wav2vec2_hindi_trt.onnx (static [1, 32000] input + embedded shape info),
because the original exports are fully dynamic with zero value_info, which
made the TRT EP refuse to partition the graph.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
trt_bootstrap = __import__("trt_bootstrap")

trt_bootstrap.setup()

import onnxruntime as ort

print("ORT version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

model_path = os.path.join(
    trt_bootstrap._PROJECT, "models", "onnx_model", "wav2vec2_hindi_trt.onnx"
)
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": os.path.join(
                trt_bootstrap._PROJECT, "onnx_optimization", "trt_cache"
            ),
        },
    ),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

try:
    sess = ort.InferenceSession(model_path, providers=providers)
    print("Session ACTIVE providers:", sess.get_providers())
    ok = "TensorrtExecutionProvider" in sess.get_providers()
    print("RESULT:", "TRT EP ACTIVE" if ok else "TRT EP NOT ACTIVE (fell back)")
except Exception as exc:  # noqa: BLE001
    print("Session creation FAILED:", exc)
    sys.exit(1)
