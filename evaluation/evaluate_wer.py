import torch
import torchaudio
import onnxruntime as ort
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import jiwer
import os

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_PATH = "models/onnx_model/wav2vec2_hindi_optimized.onnx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use synthetic test data for WER evaluation
TEST_SAMPLES = [
    ("नमस्ते", "data/sample_hindi.wav"),  # Sample audio files
]

def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy()

def greedy_decode(logits, processor):
    pred_ids = np.argmax(logits, axis=-1)
    return processor.batch_decode(pred_ids)

def evaluate_pytorch(test_samples, processor):
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    if DEVICE == "cuda":
        model = model.cuda()  # type: ignore
    model.eval()

    predictions, references = [], []

    for reference_text, audio_path in test_samples:
        waveform = load_audio(audio_path)

        inputs = processor(  # type: ignore[call-arg]
            waveform,
            sampling_rate=16000,  # type: ignore[call-arg]
            return_tensors="pt",  # type: ignore[call-arg]
            padding=True  # type: ignore[call-arg]
        )

        with torch.no_grad():
            logits = model(
                inputs.input_values.to(DEVICE)
            ).logits

        pred = greedy_decode(logits.cpu().numpy(), processor)[0]
        predictions.append(pred)
        references.append(reference_text)

    return jiwer.wer(references, predictions)

def evaluate_onnx(test_samples, processor):
    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    predictions, references = [], []

    for reference_text, audio_path in test_samples:
        waveform = load_audio(audio_path)

        inputs = processor(  # type: ignore[call-arg]
            waveform,
            sampling_rate=16000,  # type: ignore[call-arg]
            return_tensors="np",  # type: ignore[call-arg]
            padding=True  # type: ignore[call-arg]
        )

        logits = session.run(
            None, {"input_values": inputs["input_values"]}
        )[0]

        pred = greedy_decode(logits, processor)[0]
        predictions.append(pred)
        references.append(reference_text)

    return jiwer.wer(references, predictions)

def main():
    print("Using synthetic test audio for WER evaluation...")
    print("Note: Using single test sample due to dataset access constraints")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    print("Evaluating PyTorch model...")
    wer_pt = evaluate_pytorch(TEST_SAMPLES, processor)

    print("Evaluating ONNX model...")
    wer_onnx = evaluate_onnx(TEST_SAMPLES, processor)

    print("\n=== WER Results ===")
    print(f"PyTorch WER: {wer_pt:.4f}")
    print(f"ONNX WER:    {wer_onnx:.4f}")
    print("\nNote: WER=1.0 expected for synthetic audio (no real Hindi speech)")
    print("Interview point: Accuracy preservation validated in Phase 2.2 (numerical output comparison)")

if __name__ == "__main__":
    main()
