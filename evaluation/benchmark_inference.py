import time
import torch
import torchaudio
import onnxruntime as ort
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_PATH = "models/onnx_model/wav2vec2_hindi_optimized.onnx"
AUDIO_PATH = "data/sample_hindi.wav"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = 10

def load_audio():
    waveform, sr = torchaudio.load(AUDIO_PATH)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    return waveform.squeeze()

def benchmark_pytorch(waveform):
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
    input_values = inputs.input_values.to(DEVICE)

    # warmup
    with torch.no_grad():
        model(input_values)

    start = time.time()
    with torch.no_grad():
        for _ in range(RUNS):
            model(input_values)
    end = time.time()

    return (end - start) / RUNS

def benchmark_onnx(providers):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    waveform = load_audio()

    inputs = processor(
        waveform,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="np",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )

    session = ort.InferenceSession(ONNX_PATH, providers=providers)

    # warmup
    session.run(None, {"input_values": inputs["input_values"]})

    start = time.time()
    for _ in range(RUNS):
        session.run(None, {"input_values": inputs["input_values"]})
    end = time.time()

    return (end - start) / RUNS

def main():
    waveform = load_audio()

    pt_latency = benchmark_pytorch(waveform)
    onnx_latency = benchmark_onnx(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    trt_latency = benchmark_onnx(
        [
            (
                "TensorrtExecutionProvider",
                {"trt_fp16_enable": True}
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    print("=== Average Latency (seconds) ===")
    print(f"PyTorch: {pt_latency:.4f}")
    print(f"ONNX Runtime: {onnx_latency:.4f}")
    print(f"ONNX Runtime + TensorRT EP (attempt): {trt_latency:.4f}")

    print("\n=== Throughput (samples/sec) ===")
    print(f"PyTorch: {1/pt_latency:.2f}")
    print(f"ONNX Runtime: {1/onnx_latency:.2f}")
    print(f"ONNX + TensorRT EP: {1/trt_latency:.2f}")

if __name__ == "__main__":
    main()
