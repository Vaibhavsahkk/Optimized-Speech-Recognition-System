"""FP16 TensorRT EP benchmark on the static-shape TRT-ready model.

Why this file looks the way it does (all discovered by running, not guessing):
1. trt_bootstrap.setup() must run BEFORE creating the InferenceSession, or
   nvinfer_10.dll fails to load on Windows (Error 126) and ORT silently falls
   back to CPU.
2. The model is models/onnx_model/wav2vec2_hindi_trt.onnx: static [1, 32000]
   input + full embedded shape info. The original exports are fully dynamic
   with zero value_info (TRT EP refuses to partition), and the graph-optimized
   variant contains 45 com.microsoft contrib ops the TRT parser can't parse.
3. The first session.run() builds and serializes the FP16 engine into
   onnx_optimization/trt_cache (minutes on a laptop GPU); later runs load
   the cached engine.

Measured: cold-start time (engine build or cache load), then WARMUP + RUNS
timed runs -> mean/median/p95/min latency, throughput, real-time factor,
plus the greedy-CTC transcription as an accuracy sanity check.
"""
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import trt_bootstrap  # noqa: E402

trt_bootstrap.setup()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torchaudio  # noqa: E402
from transformers import Wav2Vec2Processor  # noqa: E402

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_PATH = os.path.join(_PROJECT, "models", "onnx_model", "wav2vec2_hindi_trt.onnx")
AUDIO_PATH = os.path.join(_PROJECT, "data", "sample_hindi.wav")
CACHE_PATH = os.path.join(_HERE, "trt_cache")
WARMUP = 10
RUNS = 50


def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    return waveform.squeeze().numpy()


def main():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    waveform = load_audio(AUDIO_PATH)

    inputs = processor(
        waveform,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="np",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )
    input_values = inputs["input_values"]

    print(f"model:  {ONNX_PATH}")
    print(f"audio:  {AUDIO_PATH}")
    print(f"input:  {input_values.shape} (static model expects (1, 32000))")
    if tuple(input_values.shape) != (1, 32000):
        raise ValueError(
            f"static model expects (1, 32000), got {input_values.shape}"
        )

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": CACHE_PATH,
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    print("creating session (TRT EP requested)...")
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    active = session.get_providers()
    print("active providers:", active)
    if "TensorrtExecutionProvider" not in active:
        raise RuntimeError("TRT EP not active -- aborting benchmark")

    # First run: builds the FP16 engine (or loads it from cache).
    t0 = time.perf_counter()
    logits = session.run(None, {"input_values": input_values})[0]
    cold_s = time.perf_counter() - t0
    print(f"cold start (engine build or cache load): {cold_s:.2f} s")
    print(f"logits shape: {logits.shape}")

    if os.path.isdir(CACHE_PATH):
        files = os.listdir(CACHE_PATH)
        size_mb = sum(
            os.path.getsize(os.path.join(CACHE_PATH, f)) for f in files
        ) / 1e6
        print(f"engine cache: {len(files)} file(s), {size_mb:.1f} MB")

    # Warmup to steady state, then timed runs.
    for _ in range(WARMUP):
        session.run(None, {"input_values": input_values})

    lat = []
    for _ in range(RUNS):
        t = time.perf_counter()
        session.run(None, {"input_values": input_values})
        lat.append(time.perf_counter() - t)

    lat_sorted = sorted(lat)
    mean = statistics.mean(lat)
    print(f"\n=== FP16 TensorRT latency, {RUNS} runs, 2.0 s of audio ===")
    print(f"mean:     {mean * 1000:.1f} ms")
    print(f"median:   {statistics.median(lat) * 1000:.1f} ms")
    print(f"p95:      {lat_sorted[max(int(len(lat_sorted) * 0.95) - 1, 0)] * 1000:.1f} ms")
    print(f"min:      {lat_sorted[0] * 1000:.1f} ms")
    print(f"throughput: {1 / mean:.2f} inferences/s")
    print(f"real-time factor: {2.0 / mean:.2f}x realtime")

    # Accuracy sanity check: greedy CTC decode of the same logits.
    pred_ids = np.argmax(logits, axis=-1)
    text = processor.batch_decode(pred_ids)[0]
    print(f"\ndecoded transcription: {text!r}")


if __name__ == "__main__":
    main()
