import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
ONNX_PATH = "models/onnx_model/wav2vec2_hindi.onnx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    # Dummy input (1 sec of audio @16kHz)
    dummy_audio = torch.randn(1, 16000).to(DEVICE)

    print("Exporting model to ONNX...")

    torch.onnx.export(
        model,
        dummy_audio,
        ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "audio_length"},
            "logits": {0: "batch_size", 1: "time_steps"},
        },
    )

    print(f"ONNX model exported to: {ONNX_PATH}")

if __name__ == "__main__":
    main()
