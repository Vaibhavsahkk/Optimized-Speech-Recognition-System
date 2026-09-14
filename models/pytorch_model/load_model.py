import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Using publicly available Hindi Wav2Vec2 model
# Note: ai4bharat/indicwav2vec-hindi requires gated access
MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"

def load_model():
    print("Loading processor...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    if DEVICE == "cuda":
        model = model.cuda()  # type: ignore
    model.eval()

    print("Model loaded successfully")
    return processor, model

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    return waveform.squeeze()

def run_inference(processor, model, waveform):
    # Process audio for model input
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")  # type: ignore[call-arg]

    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)

    return outputs.logits

if __name__ == "__main__":
    print("Device:", DEVICE)

    # Provide path to a single Hindi WAV file (16k or any rate)
    AUDIO_PATH = "data/sample_hindi.wav"

    processor, model = load_model()
    waveform = load_audio(AUDIO_PATH)
    logits = run_inference(processor, model, waveform)

    print("Inference successful")
    print("Logits shape:", logits.shape)
