"""Build a TensorRT-ready variant of the ONNX model.

Why this exists (all discovered by running test_trt_ep_load.py, not guessing):

1. The exported model has NO embedded shape info (value_info: 0) and a fully
   dynamic input [batch_size, audio_length]. TensorRT EP therefore refuses to
   partition the graph:
     "TensorRT input: /wav2vec2/feature_projection/layer_norm/Mul_output_0
      has no shape specified. Please run shape inference on the onnx model
      first."

2. The graph-optimized model (wav2vec2_hindi_optimized.onnx) is NOT suitable
   for TRT: it contains 45 com.microsoft contrib ops (Gelu,
   SkipLayerNormalization, ...) that the TensorRT ONNX parser cannot handle.
   So we start from the clean base export (100% standard ONNX domain, opset 14).

3. data/sample_hindi.wav is exactly 32000 frames (2.0 s @ 16 kHz), so we pin
   the input to a static [1, 32000]. Static shapes -> fully static TRT engine:
   fastest build, smallest cache, no profile-shape plumbing.

Fix: pin input dims -> run ONNX Runtime's symbolic shape inference (the exact
tool recommended by the TRT EP error/docs) -> save
models/onnx_model/wav2vec2_hindi_trt.onnx and verify the previously-failing
tensor now carries a shape.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import onnx  # noqa: E402
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference  # noqa: E402

BASE_ONNX = os.path.join(_PROJECT, "models", "onnx_model", "wav2vec2_hindi.onnx")
OUT_ONNX = os.path.join(_PROJECT, "models", "onnx_model", "wav2vec2_hindi_trt.onnx")

# data/sample_hindi.wav = 32000 frames @ 16 kHz = 2.0 s
STATIC_INPUT = [1, 32000]

# The tensor TRT previously choked on -- must have a shape after inference.
WATCH_TENSOR = "/wav2vec2/feature_projection/layer_norm/Mul_output_0"


def dims_of(value_info):
    return [
        d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim
    ]


def main():
    print(f"Loading base model: {BASE_ONNX}")
    m = onnx.load(BASE_ONNX)

    # 1) Pin the input to a static [1, 32000].
    #    dim_param/dim_value live in a protobuf oneof, so assigning dim_value
    #    automatically clears the dynamic dim_param.
    inp = next(i for i in m.graph.input if i.name == "input_values")
    dims = inp.type.tensor_type.shape.dim
    if len(dims) != len(STATIC_INPUT):
        raise ValueError(f"unexpected input rank: {dims_of(inp)}")
    for dim, value in zip(dims, STATIC_INPUT):
        dim.dim_value = value
    print("Input pinned to:", STATIC_INPUT)

    # 2) Symbolic shape inference (ORT tool recommended by the TRT EP docs).
    #    auto_merge=True lets conflicting symbolic dims merge instead of erroring.
    print("Running symbolic shape inference (auto_merge=True)...")
    inferred = SymbolicShapeInference.infer_shapes(m, auto_merge=True)
    if inferred is None:
        raise RuntimeError("shape inference returned None (opset too low?)")

    vi = {v.name: v for v in inferred.graph.value_info}
    print(f"value_info entries: {len(vi)} (was 0)")

    if WATCH_TENSOR not in vi:
        raise RuntimeError(f"watch tensor {WATCH_TENSOR} still has no shape info")
    print(f"OK watch tensor now shaped: {WATCH_TENSOR} -> {dims_of(vi[WATCH_TENSOR])}")

    # Sanity: input/output signatures after inference.
    for i in inferred.graph.input:
        print("INPUT", i.name, dims_of(i))
    for o in inferred.graph.output:
        print("OUTPUT", o.name, dims_of(o))

    # 3) Save (weights stay embedded; ~360 MB).
    print(f"Saving: {OUT_ONNX}")
    onnx.save(inferred, OUT_ONNX)
    print("Done. Next: point TensorrtExecutionProvider at wav2vec2_hindi_trt.onnx.")


if __name__ == "__main__":
    main()
