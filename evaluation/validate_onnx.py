"""Phase 2.2: numerical validation -- ONNX Runtime vs PyTorch logits.

The DLL bootstrap (torch first, then cu12 wheels) is REQUIRED before creating
any InferenceSession, or the CUDA EP silently falls back to CPU on this Windows
dev box (cublasLt64_12.dll lives in the nvidia-* pip wheels, not on PATH).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT, "onnx_optimization"))

import trt_bootstrap  # noqa: E402

trt_bootstrap.setup(verbose=False)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # noqa: E402

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_PATH = os.path.join(_PROJECT, "models", "onnx_model", "wav2vec2_hindi.onnx")
AUDIO_PATH = os.path.join(_PROJECT, "data", "sample_hindi.wav")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    return waveform.squeeze()

def pytorch_inference(waveform):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
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

def onnx_inference(waveform):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    inputs = processor(
        waveform,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="np",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )

    sess = ort.InferenceSession(
        ONNX_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    logits = sess.run(
        None,
        {"input_values": inputs["input_values"]}
    )[0]

    return logits

def main():
    waveform = load_audio(AUDIO_PATH)

    torch_out = pytorch_inference(waveform)
    onnx_out = onnx_inference(waveform)

    diff = np.max(np.abs(torch_out - onnx_out))
    mean_diff = np.mean(np.abs(torch_out - onnx_out))

    print("Max absolute difference:", diff)
    print("Mean absolute difference:", mean_diff)

    # Argmax agreement == identical greedy CTC decoding.
    torch_ids = np.argmax(torch_out, axis=-1)
    onnx_ids = np.argmax(onnx_out, axis=-1)
    agree = float((torch_ids == onnx_ids).mean()) * 100
    print(f"argmax agreement (greedy CTC tokens): {agree:.1f}%")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    print("transcriptions identical:",
          processor.batch_decode(torch_ids)[0] == processor.batch_decode(onnx_ids)[0])

if __name__ == "__main__":
    main()
