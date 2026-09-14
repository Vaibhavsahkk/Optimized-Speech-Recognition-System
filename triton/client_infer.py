import numpy as np
import torchaudio
import tritonclient.http as httpclient
from transformers import Wav2Vec2Processor

TRITON_URL = "localhost:8000"
MODEL_NAME = "wav2vec2_hindi"
AUDIO_PATH = "../data/sample_hindi.wav"

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
        audio,
        sampling_rate=16000,  # type: ignore[call-arg]
        return_tensors="np",  # type: ignore[call-arg]
        padding=True  # type: ignore[call-arg]
    )

    client = httpclient.InferenceServerClient(url=TRITON_URL)

    triton_inputs = [
        httpclient.InferInput(
            "input_values",
            inputs["input_values"].shape,
            "FP32"
        )
    ]
    triton_inputs[0].set_data_from_numpy(inputs["input_values"])

    triton_outputs = [
        httpclient.InferRequestedOutput("logits")
    ]

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=triton_inputs,
        outputs=triton_outputs
    )

    logits = response.as_numpy("logits")
    pred_ids = np.argmax(logits, axis=-1)  # type: ignore[call-overload]
    transcription = processor.batch_decode(pred_ids)

    print("Transcription:", transcription[0])

if __name__ == "__main__":
    main()
