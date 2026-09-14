"""Verify the CUDA EP is actually ACTIVE now (after nvidia_ort_loader fix)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
trt_bootstrap = __import__("trt_bootstrap")

import nvidia_ort_loader

nvidia_ort_loader.setup()

import numpy as np
import onnxruntime as ort

print("ORT:", ort.__version__)
model = os.path.join(
    trt_bootstrap._PROJECT, "models", "onnx_model", "wav2vec2_hindi_optimized.onnx"
)
sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ACTIVE providers:", sess.get_providers())
x = np.zeros((1, 32000), dtype=np.float32)
out = sess.run(None, {"input_values": x})
print("run OK, output shape:", out[0].shape)
print("RESULT:", "CUDA EP ACTIVE" if sess.get_providers()[0] == "CUDAExecutionProvider" else "FELL BACK TO CPU")
