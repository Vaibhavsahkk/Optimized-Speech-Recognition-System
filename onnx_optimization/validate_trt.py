"""Accuracy check: compare FP16 TensorRT EP logits against the PyTorch
FP32 baseline on data/sample_hindi.wav.

Why: performance must be weighed against accuracy. FP16
execution trades some numeric precision for the 3.1x speedup, so we verify
the greedy CTC decoding is unchanged and quantify the logit deviation.

Companion to validate_onnx.py (which does the same for the FP32 ONNX path).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # noqa: E402

import trt_bootstrap  # noqa: E402

trt_bootstrap.setup(verbose=False)

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
TRT_ONNX = os.path.join(_PROJECT, "models", "onnx_model", "wav2vec2_hindi_trt.onnx")
AUDIO_PATH = os.path.join(_PROJECT, "data", "sample_hindi.wav")
CACHE_PATH = os.path.join(_HERE, "trt_cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    return waveform.squeeze().numpy()


def pytorch_inference(processor, waveform):
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

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE)).logits

    return logits.cpu().numpy()


def trt_inference(processor, waveform):
    inputs = processor(
        waveform,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="np",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )

    sess = ort.InferenceSession(
        TRT_ONNX,
        providers=[
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
        ],
    )

    active = sess.get_providers()
    if "TensorrtExecutionProvider" not in active:
        raise RuntimeError(f"TRT EP not active: {active}")

    return sess.run(None, {"input_values": inputs["input_values"]})[0]


def main():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    waveform = load_audio(AUDIO_PATH)

    print("running PyTorch FP32 baseline...")
    torch_logits = pytorch_inference(processor, waveform)

    print("running TensorRT EP FP16...")
    trt_logits = trt_inference(processor, waveform)

    diff = np.abs(torch_logits - trt_logits)
    print("\n=== FP16 TRT vs PyTorch FP32 (logit deviation) ===")
    print("max absolute difference:", float(diff.max()))
    print("mean absolute difference:", float(diff.mean()))
    print("logits range (torch):", float(torch_logits.min()), float(torch_logits.max()))
    print("logits range (trt):  ", float(trt_logits.min()), float(trt_logits.max()))

    # Argmax agreement = identical greedy CTC decoding.
    torch_ids = np.argmax(torch_logits, axis=-1)
    trt_ids = np.argmax(trt_logits, axis=-1)
    agree = float((torch_ids == trt_ids).mean()) * 100
    print(f"\nargmax agreement (greedy CTC tokens): {agree:.1f}%")

    torch_text = processor.batch_decode(torch_ids)[0]
    trt_text = processor.batch_decode(trt_ids)[0]
    print("torch transcription:", repr(torch_text))
    print("trt   transcription:", repr(trt_text))
    print("\nidentical transcription:", torch_text == trt_text)


if __name__ == "__main__":
    main()
