# ASR Inference Optimization with ONNX Runtime, TensorRT & Triton

## Overview
This project demonstrates an end-to-end **MLOps-focused optimization pipeline** for a Hindi Automatic Speech Recognition (ASR) system based on **Wav2Vec2**.

The goal is to:
- Profile model inference
- Convert and optimize the model using ONNX
- Leverage TensorRT acceleration (via ONNX Runtime)
- Benchmark performance vs accuracy
- Design a production-ready deployment using **NVIDIA Triton Inference Server**

The project follows **real-world production constraints**, especially Windows vs Linux deployment differences.

---

## Model
- Architecture: Wav2Vec2 (CTC-based ASR)
- Language: Hindi
- Model Used: `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`
- Frameworks: PyTorch → ONNX → ONNX Runtime → TensorRT EP

---

## Project Structure
```
.
├── models/
│   ├── pytorch_model/
│   └── onnx_model/
├── profiling/
├── onnx_export/
├── onnx_optimization/
├── evaluation/
├── triton/
│   ├── model_repository/
│   ├── model_analyzer/
│   └── PRODUCTION_DEPLOYMENT.md
└── README.md
```

Large model, audio, log, and profiler artifacts are intentionally excluded from
version control. Run the export scripts to recreate model artifacts locally.

---

## Phase-wise Breakdown

### Phase 0 — Environment Setup
- Python 3.10 (isolated venv)
- CUDA-enabled PyTorch
- ONNX Runtime (GPU)
- TensorRT Execution Provider
- Triton client & Model Analyzer tools

---

### Phase 1 — Model Loading & Profiling
- Loaded Hindi Wav2Vec2 ASR model
- Ran GPU inference
- Identified bottlenecks using **PyTorch Profiler**
  - GEMM (linear layers)
  - Conv1D feature extractor

---

### Phase 2 — ONNX Export & Validation
- Exported model to ONNX (opset 14)
- Enabled dynamic axes for variable-length audio
- Validated ONNX vs PyTorch outputs (numerical equivalence)

---

### Phase 3 — Optimization & TensorRT
- Applied ONNX Runtime graph optimizations
- Enabled FP16 execution via TensorRT Execution Provider
- Documented Windows limitations and fallback behavior

---

### Phase 4 — Performance Benchmarking
Benchmarked:
- PyTorch (baseline)
- ONNX Runtime
- ONNX Runtime + TensorRT EP

Observed trade-offs between runtime overhead and kernel efficiency for batch size = 1.

---

### Phase 5 — Accuracy Evaluation
- Evaluated WER using greedy CTC decoding
- Verified accuracy preservation via numerical output comparison
- Documented dataset access constraints and production assumptions

---

### Phase 6 — Triton Deployment Design
- Designed Triton model repository
- Created `config.pbtxt` with dynamic batching
- Implemented Triton client inference workflow
- Designed Model Analyzer configuration
- Documented Linux/Docker production deployment strategy

---

## Production Deployment
See: `triton/PRODUCTION_DEPLOYMENT.md`

Production is designed for:
- Linux
- Docker
- NVIDIA Triton Inference Server
- TensorRT-accelerated inference

Windows was used for development and validation only.

---

## Key Learnings
- Profiling before optimization is critical
- ONNX export requires careful opset and dynamic axis handling
- TensorRT acceleration is environment-dependent
- Production MLOps requires clear separation of dev vs prod constraints

---

## Future Improvements
- Batch size > 1 benchmarking
- INT8 quantization with calibration
- Streaming ASR support
- Full Triton Model Analyzer runs on Linux
