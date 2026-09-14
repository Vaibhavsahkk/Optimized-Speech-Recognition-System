import time
import onnxruntime as ort
import torchaudio
from transformers import Wav2Vec2Processor

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_PATH = "models/onnx_model/wav2vec2_hindi_optimized.onnx"
AUDIO_PATH = "data/sample_hindi.wav"

def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    return waveform.squeeze()

def main():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    waveform = load_audio(AUDIO_PATH)

    inputs = processor(
        waveform,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="np",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache"
            }
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(
        ONNX_PATH,
        providers=providers
    )

    # Warm-up
    for _ in range(3):
        _ = session.run(None, {"input_values": inputs["input_values"]})

    # Timed inference
    start = time.time()
    outputs = session.run(None, {"input_values": inputs["input_values"]})
    end = time.time()

    print("Inference completed")
    print("Latency (seconds):", end - start)
    print("Output shape:", outputs[0].shape)  # type: ignore[union-attr]

if __name__ == "__main__":
    main()
