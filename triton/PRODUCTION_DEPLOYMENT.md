# Production Deployment Plan - Triton Inference Server

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
docker pull nvcr.io/nvidia/tritonserver:25.03-py3
```

### Run Triton Server
```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.03-py3 \
  tritonserver --model-repository=/models
```

### Model Repository
The model repository follows Triton conventions:

```
model_repository/
  wav2vec2_hindi/
    config.pbtxt
    1/
      model.onnx
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
Windows was used for development, validation, and benchmarking - including a
**verified working TensorRT EP FP16 path** (see `onnx_optimization/` and the
README benchmark table: 2.97x vs PyTorch, 100% argmax agreement). Triton Server
**was also verified live on this Windows dev box** via Docker Desktop WSL2 GPU
passthrough (image `25.03-py3`, model READY on GPU, HTTP serving benchmark:
13.6 ms mean / 73.4 inf/s for 2 s audio, batch 1 - see
`triton/bench_serving.py`). For production, the Linux deployment below remains
the recommended path.

---

## Verified Local Serving Run (Windows, WSL2 GPU passthrough)

```powershell
docker run -d --name triton_wav2vec2 --gpus all `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}\triton\model_repository:/models" `
  nvcr.io/nvidia/tritonserver:25.03-py3 `
  tritonserver --model-repository=/models

# then from triton/:
#   python client_infer.py     # transcription over HTTP
#   python bench_serving.py     # end-to-end HTTP latency benchmark
```

Observed: model READY on GPU device 0 in ~6 s; live HTTP inference verified
(identical transcription to local runs); 13.6 ms mean / 14.8 ms p95 over HTTP.

---

## Deployment Rationale
**Summary:**
> "Development and validation were done on Windows. Production deployment is designed for Linux using Dockerized Triton Server, which provides stable TensorRT acceleration, Model Analyzer support, and scalable inference."

**Technical Justification:**
- Triton Server is Linux-first with full GPU driver support
- TensorRT libraries integrate seamlessly in containerized environments
- Docker enables reproducible deployments across infrastructure
- Model Analyzer provides automated performance tuning
