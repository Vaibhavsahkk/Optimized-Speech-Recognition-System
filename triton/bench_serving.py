"""Measure end-to-end serving latency through Triton's HTTP endpoint.

Complements evaluation/benchmark_inference.py: that measures pure runtime
compute; this adds the full serving stack (HTTP serialization + Triton
request handling + GPU inference). Run with the Triton container up:
    docker run -d --gpus all -p 8000:8000 -v <repo>:/models \
        nvcr.io/nvidia/tritonserver:25.03-py3 tritonserver --model-repository=/models
"""
import time

import numpy as np
import torchaudio
import tritonclient.http as httpclient
from transformers import Wav2Vec2Processor

TRITON_URL = "localhost:8000"
MODEL_NAME = "wav2vec2_hindi"
AUDIO_PATH = "../data/sample_hindi.wav"
N_WARMUP = 5
N_REQUESTS = 30


def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze().numpy()


def main():
    processor = Wav2Vec2Processor.from_pretrained(
        "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
    )
    audio = load_audio(AUDIO_PATH)
    inputs = processor(  # type: ignore[call-arg]
        audio, sampling_rate=16000, return_tensors="np", padding=True
    )

    client = httpclient.InferenceServerClient(url=TRITON_URL)

    def infer_once():
        t_in = httpclient.InferInput("input_values", inputs["input_values"].shape, "FP32")
        t_in.set_data_from_numpy(inputs["input_values"])
        out = httpclient.InferRequestedOutput("logits")
        resp = client.infer(model_name=MODEL_NAME, inputs=[t_in], outputs=[out])
        return resp.as_numpy("logits")

    for _ in range(N_WARMUP):
        infer_once()

    lat = []
    for _ in range(N_REQUESTS):
        t0 = time.perf_counter()
        logits = infer_once()
        lat.append((time.perf_counter() - t0) * 1000.0)

    lat_sorted = sorted(lat)
    print(f"Triton HTTP serving latency ({N_REQUESTS} requests, "
          f"{len(audio)/16000:.1f} s audio, batch 1):")
    print(f"  mean   : {np.mean(lat):6.1f} ms")
    print(f"  median : {lat_sorted[len(lat)//2]:6.1f} ms")
    print(f"  p95    : {lat_sorted[int(len(lat)*0.95)-1]:6.1f} ms")
    print(f"  min    : {lat_sorted[0]:6.1f} ms")
    print(f"  through: {N_REQUESTS / (sum(lat)/1000):.1f} inf/s over HTTP")
    pred = processor.batch_decode(np.argmax(logits, axis=-1))
    print("transcription:", pred[0])


if __name__ == "__main__":
    main()
