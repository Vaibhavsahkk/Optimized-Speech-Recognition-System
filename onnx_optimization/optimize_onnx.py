import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer

INPUT_ONNX = "models/onnx_model/wav2vec2_hindi.onnx"
OUTPUT_ONNX = "models/onnx_model/wav2vec2_hindi_optimized.onnx"

def main():
    print("Loading ONNX model...")
    model = onnx.load(INPUT_ONNX)

    print("Running ONNX Runtime graph optimizations...")
    optimized_model = optimizer.optimize_model(
        INPUT_ONNX,
        model_type="bert",   # closest supported transformer class
        num_heads=12,        # safe default for Wav2Vec2-style encoders
        hidden_size=768      # typical Wav2Vec2 hidden size
    )

    optimized_model.save_model_to_file(OUTPUT_ONNX)
    print(f"Optimized ONNX model saved to: {OUTPUT_ONNX}")

if __name__ == "__main__":
    main()
