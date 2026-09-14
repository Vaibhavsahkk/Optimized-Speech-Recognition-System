# Production Deployment Plan — Triton Inference Server

## Overview
Local development was performed on Windows using ONNX Runtime and TensorRT Execution Provider.
For production deployment, the system is designed to run on Linux using Dockerized Triton Inference Server.

---

## Target Environment
- OS: Ubuntu 20.04 / 22.04
- GPU: NVIDIA GPU with TensorRT support
- CUDA: 11.8+
- Docker + NVIDIA Container Toolkit

---

## Triton Server Deployment

### Pull Triton Server Image
```bash
docker pull nvcr.io/nvidia/tritonserver:23.10-py3
```

### Run Triton Server
```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### Model Repository
The model repository follows Triton conventions:

```
model_repository/
└── wav2vec2_hindi/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

---

## Triton Client
Client applications use HTTP or gRPC to:
- Send preprocessed audio tensors
- Receive logits
- Perform decoding client-side

---

## Model Analyzer
Model Analyzer is executed inside the Triton container to:
- Evaluate latency vs throughput
- Tune batch sizes and concurrency
- Identify optimal serving configuration

### Run Model Analyzer
```bash
model-analyzer profile \
  --model-repository ./model_repository \
  --config-file ./model_analyzer/model_analyzer_config.yaml \
  --output-model-repository-path ./model_analyzer_output
```

---

## Why Linux/Docker?
- Native TensorRT support
- Stable Triton Server execution
- Production-grade GPU scheduling
- Industry-standard deployment pattern

---

## Windows Development Note
Windows was used for development and validation.
Production inference is explicitly designed for Linux/Docker,
which reflects real-world MLOps practices.

---

## Interview Points
**Key Explanation:**
> "Development and validation were done on Windows. Production deployment is designed for Linux using Dockerized Triton Server, which provides stable TensorRT acceleration, Model Analyzer support, and scalable inference."

**Technical Justification:**
- Triton Server is Linux-first with full GPU driver support
- TensorRT libraries integrate seamlessly in containerized environments
- Docker enables reproducible deployments across infrastructure
- Model Analyzer provides automated performance tuning
