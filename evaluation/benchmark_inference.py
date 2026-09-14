"""Latency/throughput benchmark -- PyTorch vs ONNX Runtime (CUDA EP)
vs ONNX Runtime + TensorRT EP (FP16).

Model selection per runtime (discovered by running, not guessing):
- PyTorch:      HF model directly (baseline).
- ONNX Runtime: models/onnx_model/wav2vec2_hindi_optimized.onnx (graph
                optimizations + CUDA EP; contrib ops are fine for ORT).
- TensorRT EP:  models/onnx_model/wav2vec2_hindi_trt.onnx (static [1, 32000]
                input + embedded shape info; TRT needs both, and cannot parse
                the com.microsoft contrib ops in the optimized graph).

trt_bootstrap.setup() must run before any InferenceSession or the TRT EP
silently falls back (missing nvinfer_10.dll on Windows).
"""
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_ONNX_OPT_DIR = os.path.join(_PROJECT, "onnx_optimization")
sys.path.insert(0, _ONNX_OPT_DIR)

import trt_bootstrap  # noqa: E402

trt_bootstrap.setup()

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: E402

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_OPTIMIZED = os.path.join(
    _PROJECT, "models", "onnx_model", "wav2vec2_hindi_optimized.onnx"
)
ONNX_TRT = os.path.join(_PROJECT, "models", "onnx_model", "wav2vec2_hindi_trt.onnx")
AUDIO_PATH = os.path.join(_PROJECT, "data", "sample_hindi.wav")
TRT_CACHE = os.path.join(_ONNX_OPT_DIR, "trt_cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 10
RUNS = 50


def load_audio():
    waveform, sr = torchaudio.load(AUDIO_PATH)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    return waveform.squeeze()


def timed(fn, feed) -> dict:
    for _ in range(WARMUP):
        fn(feed)

    lat = []
    for _ in range(RUNS):
        t = time.perf_counter()
        fn(feed)
        lat.append(time.perf_counter() - t)

    lat.sort()
    mean = statistics.mean(lat)
    return {
        "mean": mean,
        "median": statistics.median(lat),
        "p95": lat[max(int(len(lat) * 0.95) - 1, 0)],
        "min": lat[0],
    }


def benchmark_pytorch(waveform, processor):
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    if DEVICE == "cuda":
        model = model.cuda()  # type: ignore
    model.eval()

    inputs = processor(
        waveform,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="pt",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        stats = timed(lambda _: model(input_values), None)

    del model, input_values
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return stats


def benchmark_ort(waveform, processor, model_path, providers):
    inputs = processor(
        waveform,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="np",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )
    feed = {"input_values": inputs["input_values"]}

    session = ort.InferenceSession(model_path, providers=providers)
    stats = timed(lambda f: session.run(None, f), feed)

    del session
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return stats


def main():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    waveform = load_audio()

    print(f"device: {DEVICE}, runs: {RUNS}, audio: 2.0 s (32000 frames)\n")

    print("benchmarking PyTorch...")
    pt = benchmark_pytorch(waveform, processor)

    print("benchmarking ONNX Runtime (CUDA EP, optimized model)...")
    onnx = benchmark_ort(
        waveform, processor, ONNX_OPTIMIZED, ["CUDAExecutionProvider"]
    )

    print("benchmarking ONNX Runtime + TensorRT EP (FP16, static model)...")
    trt = benchmark_ort(
        waveform,
        processor,
        ONNX_TRT,
        [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": TRT_CACHE,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    rows = [
        ("PyTorch (baseline)", pt),
        ("ONNX Runtime (CUDA EP)", onnx),
        ("ONNX Runtime + TRT EP (FP16)", trt),
    ]

    print("\n=== Average latency (ms), 2.0 s audio, batch=1 ===")
    print(f"{'runtime':<32}{'mean':>9}{'median':>9}{'p95':>9}{'min':>9}")
    for name, s in rows:
        print(
            f"{name:<32}{s['mean'] * 1000:>9.1f}{s['median'] * 1000:>9.1f}"
            f"{s['p95'] * 1000:>9.1f}{s['min'] * 1000:>9.1f}"
        )

    print("\n=== Throughput (inferences/s) ===")
    for name, s in rows:
        print(f"{name:<32}{1 / s['mean']:>8.2f}")

    print("\n=== Real-time factor (audio-sec processed per wall-sec) ===")
    for name, s in rows:
        print(f"{name:<32}{2.0 / s['mean']:>8.1f}x")

    fastest = min(s["mean"] for _, s in rows)
    print("\n=== Speedup vs PyTorch baseline ===")
    for name, s in rows:
        print(f"{name:<32}{pt['mean'] / s['mean']:>8.2f}x  (vs fastest: {fastest / s['mean']:.2f}x)")


if __name__ == "__main__":
    main()
