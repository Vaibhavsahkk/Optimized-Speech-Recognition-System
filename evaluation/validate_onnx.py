import torch
import torchaudio
import onnxruntime as ort
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_PATH = "models/onnx_model/wav2vec2_hindi.onnx"
AUDIO_PATH = "data/sample_hindi.wav"

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

if __name__ == "__main__":
    main()
