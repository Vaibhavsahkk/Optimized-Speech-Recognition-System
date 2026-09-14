import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
AUDIO_PATH = "data/sample_hindi.wav"

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    return waveform.squeeze()

def main():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    if DEVICE == "cuda":
        model = model.cuda()  # type: ignore
    model.eval()

    waveform = load_audio(AUDIO_PATH)

    # Process audio for model input
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")  # type: ignore[call-arg]
    input_values = inputs.input_values.to(DEVICE)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(input_values)

    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15
    ))

if __name__ == "__main__":
    main()
