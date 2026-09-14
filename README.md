# Hindi Wav2Vec2 ASR Inference Optimization

## Abstract

This repository contains a complete inference optimization and serving
pipeline for a Hindi automatic speech recognition model. The system takes a
Wav2Vec2 CTC model from PyTorch through ONNX export, ONNX Runtime graph
optimization, TensorRT FP16 execution, and deployment behind NVIDIA Triton
Inference Server. Every optimization step in the pipeline is verified by
numerical comparison against the PyTorch baseline, and every performance
claim in this document was measured by running the scripts in this
repository on the hardware described below.

## Model

The model is a Wav2Vec2 encoder with a CTC head for Hindi,
`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` from Hugging Face, with
67 output classes. The model is consumed in three forms: the original
PyTorch weights, an ONNX export with dynamic batch and time axes, and a
static shape variant prepared specifically for TensorRT. Input is 16 kHz
mono audio; output is a frame level logit tensor that is decoded greedily
with the model tokenizer.

## Verified Environment

All measurements in this document were produced on the following
environment:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB) |
| GPU driver | 566.07 |
| OS | Windows 11, development on native Windows, Triton in Docker Desktop WSL2 |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu118 |
| Transformers | 5.0.0 |
| ONNX Runtime GPU | 1.23.2 |
| TensorRT | 10.13.3.9.post1 |
| Triton Inference Server | 25.03 py3 container |
| tritonclient | 2.64.0 |

The model itself is downloaded from Hugging Face on first use and cached
under the user cache directory. Model binaries, audio samples, engine
caches, and captured logs are intentionally not stored in version control.
Running the scripts regenerates every artifact locally.

## Repository Layout

The repository is organized by pipeline stage. Each directory contains the
script for one stage together with a captured output file for that stage.

```
data/                       Synthetic 2 s 16 kHz test signal generation
profiling/                  PyTorch profiler stage
models/pytorch_model/        PyTorch baseline loading and inference
onnx_export/                ONNX export
onnx_optimization/          Graph optimization, TensorRT preparation,
                            Windows DLL bootstrap, provider diagnostics,
                            FP16 validation
evaluation/                 Latency and throughput benchmark, ONNX
                            numerical validation, WER evaluation
triton/                     Triton model repository, HTTP clients,
                            serving benchmark, production deployment guide
```

The named scripts inside the optimization and evaluation directories:

```
onnx_optimization/
    optimize_onnx.py            ONNX Runtime graph optimization
    make_trt_ready_model.py     Static shape variant for TensorRT
    trt_bootstrap.py            Windows TensorRT DLL bootstrap
    nvidia_ort_loader.py        Windows CUDA DLL preloader
    test_cuda_ep.py             CUDA provider activation diagnostic
    test_trt_ep_load.py         TensorRT provider activation diagnostic
    validate_trt.py             FP16 numerical validation
    run_fp16_trt.py             Standalone FP16 latency benchmark

evaluation/
    benchmark_inference.py      Three runtime latency benchmark
    validate_onnx.py            ONNX vs PyTorch numerical validation
    evaluate_wer.py             Word error rate evaluation
```


## Pipeline

### Profiling

`profiling/torch_profiler.py` profiles the PyTorch baseline on GPU with
the torch profiler, recording CPU and CUDA activity with shapes and
memory. The recorded profile shows inference cost concentrated in the
transformer encoder matrix multiplications and the convolutional feature
extractor, which establishes the optimization targets for the ONNX and
TensorRT work. `profiling/profiler_output.txt` contains the captured
profile table.

### ONNX Export

`onnx_export/export_to_onnx.py` exports the model with `torch.onnx.export`
at opset 14, which is the version required for the attention related
operators in this graph. Both the batch and audio length axes are exported
as dynamic so a single export serves variable length audio. Correctness of
the export is not assumed: `evaluation/validate_onnx.py` compares ONNX
Runtime outputs against the PyTorch model on the test signal and is part
of the standard verification flow.

### ONNX Runtime Optimization

`onnx_optimization/optimize_onnx.py` applies the ONNX Runtime transformers
optimizer to the base export. The optimizer fuses attention, layer
normalization, and Gelu subgraphs, and folds constants, producing
`wav2vec2_hindi_optimized.onnx`. This optimized graph is the artifact used
by the ONNX Runtime CUDA execution provider benchmark and by the Triton
model repository.

### TensorRT Preparation

The graph optimized artifact cannot be consumed by TensorRT: it contains 45
`com.microsoft` contributed operators that the TensorRT ONNX parser does not
implement. `onnx_optimization/make_trt_ready_model.py` therefore builds
`wav2vec2_hindi_trt.onnx` from the clean base export instead. The script
fixes the input to a static `[1, 32000]` shape for the 2 second test signal
and runs ONNX Runtime symbolic shape inference to embed tensor shapes in
the graph. The fully static, fully standard domain graph allows the

### Windows DLL Bootstrap

The development machine runs PyTorch built for CUDA 11.8 and ONNX Runtime
GPU built for CUDA 12 in the same process. Windows does not search pip
wheel directories for dependent DLLs, so without explicit preloading the
CUDA and TensorRT providers silently fall back to the CPU. Two small
modules solve this. `onnx_optimization/nvidia_ort_loader.py` imports torch
first so its cuDNN 9 owns the process wide name, then preloads the cu12
cublas, cudart, cufft, and nvrtc DLLs from the nvidia wheels for ONNX
Runtime. `onnx_optimization/trt_bootstrap.py` then loads the TensorRT 10
runtime DLLs in dependency order. Every benchmark and validation script
calls `trt_bootstrap.setup()` before creating a session. The two
diagnostic scripts in the same directory print the active provider list
of a session and are used to confirm that the CUDA and TensorRT providers
are actually active rather than silently degraded.

## Performance

`evaluation/benchmark_inference.py` measures latency for the PyTorch
baseline, ONNX Runtime with the CUDA execution provider on the optimized
graph, and ONNX Runtime with the TensorRT execution provider in FP16 on
the static graph. The benchmark uses the 2 second test signal, batch size
one, 10 warmup iterations, and 50 measured iterations.

| Runtime | Mean | Median | P95 | Min | Throughput | Speedup |
|---|---|---|---|---|---|---|
| PyTorch FP32 baseline | 13.2 ms | 9.9 ms | 16.3 ms | 8.5 ms | 75.9 inf/s | 1.00x |
| ONNX Runtime CUDA EP FP32 | 10.0 ms | 9.9 ms | 10.3 ms | 9.6 ms | 100.4 inf/s | 1.32x |
| ONNX Runtime TRT EP FP16 | 4.4 ms | 4.5 ms | 4.9 ms | 3.8 ms | 225.2 inf/s | 2.97x |

`onnx_optimization/run_fp16_trt.py` is a standalone FP16 benchmark over 50
runs on the same signal: 4.4 ms mean, 4.6 ms P95, 228.5 inf/s, 457x
realtime, with a warm engine cache load of 0.12 s from a 202.9 MB cache.
FP16 execution was confirmed active from the session provider list and
from the TensorRT engine cache being populated.

TensorRT execution provider to partition and offload the encoder without

## Accuracy

Accuracy preservation is established numerically rather than by visual
inspection, at every conversion step.

`onnx_optimization/validate_trt.py` compares TensorRT EP FP16 logits
against the PyTorch FP32 baseline on the test signal. The maximum absolute
logit difference is 0.466 and the mean absolute difference is 0.134 over a
logit range of approximately negative 27 to positive 9. The argmax token
sequence agrees on 100% of frames and both paths produce the identical
greedy CTC transcription.

`evaluation/validate_onnx.py` compares ONNX Runtime FP32 against PyTorch
FP32. The maximum absolute difference is 0.416 and the mean absolute
difference is 0.122. The residual difference is consistent with TF32
GEMM arithmetic on Ampere class GPUs. Argmax agreement is 100% and the
transcriptions are identical.

`evaluation/evaluate_wer.py` computes the word error rate of the PyTorch
and ONNX paths against a reference. Both report 1.0000. The bundled test
signal is a 440 Hz synthetic tone with no speech content, so a full
substitution error is the mathematically expected result for a model
transcribing a signal containing no words. A real speech word error rate
requires an evaluation corpus with real Hindi speech; Common Voice Hindi,
the standard corpus, is access gated on Hugging Face. Accuracy
preservation through the conversion pipeline is therefore established by
the logit and argmax comparisons above. To evaluate on real speech, place
a 16 kHz mono Hindi WAV in `data/` and extend the `TEST_SAMPLES` list in
`evaluation/evaluate_wer.py` with the file and its reference text.

## Triton Serving

The production serving path is NVIDIA Triton Inference Server. The model
repository under `triton/model_repository/wav2vec2_hindi/` configures the
optimized ONNX graph on the `onnxruntime_onnx` platform with GPU instances
and dynamic batching for preferred batch sizes 1, 4, and 8. The Triton
ONNX Runtime backend accepts the contributed operators in the optimized
graph, which is why this artifact, and not the TensorRT variant, is the
served model.

Serving was verified live with Triton 25.03 running in Docker Desktop
with WSL2 GPU passthrough on the same development machine. The server
reports the model READY on GPU device 0 within approximately 6 seconds of
startup. `triton/client_infer.py` performs HTTP inference against the
running server and reproduces the transcription obtained from the local
validation scripts, confirming that the served model and the validated
model produce identical output. `triton/bench_serving.py` measures the
complete HTTP round trip: 30 sequential requests, 2 second audio, batch
size 1, resulting in a mean of 13.6 ms, median of 11.3 ms, and P95 of
14.8 ms, at 73.4 inferences per second. The gap between this figure and
the 4.4 ms compute latency of the fastest runtime is HTTP transport and
server request handling, which is the honest accounting for an end to
end served request on this hardware.

`triton/PRODUCTION_DEPLOYMENT.md` documents the production deployment:

## Reproduction

The pipeline is sequential. Each step assumes the outputs of the previous
steps exist under `models/` and `data/`.

```powershell
# One time environment setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 transformers==5.0.0 onnxruntime-gpu==1.23.2 tensorrt-cu12==10.13.3.9.post1 tritonclient==2.64.0 numpy soundfile jiwer
```

```powershell
# Generate the 2 s synthetic test signal
python data\generate_sample.py

# PyTorch baseline and profiler
python models\pytorch_model\load_model.py
python profiling\torch_profiler.py

# ONNX export and optimization
python onnx_export\export_to_onnx.py
python onnx_optimization\optimize_onnx.py
python onnx_optimization\make_trt_ready_model.py

# Verify providers are active (Windows)
python onnx_optimization\test_cuda_ep.py
python onnx_optimization\test_trt_ep_load.py

# Validation and benchmarks
python evaluation\validate_onnx.py
python onnx_optimization\validate_trt.py
python onnx_optimization\run_fp16_trt.py
python evaluation\benchmark_inference.py
python evaluation\evaluate_wer.py

# Triton serving
docker run -d --name triton_wav2vec2 --gpus all `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "<repo path>\triton\model_repository:/models" `
  nvcr.io/nvidia/tritonserver:25.03-py3 `

## Results Summary

| Stage | Latency, mean | Result |
|---|---|---|
| PyTorch FP32 baseline | 13.2 ms | Baseline |
| ONNX Runtime CUDA EP FP32 | 10.0 ms | 100% argmax agreement vs PyTorch |
| TensorRT EP FP16 | 4.4 ms | 100% argmax agreement, identical transcription |
| Triton served over HTTP | 13.6 ms | End to end, identical transcription |

The pipeline delivers a 2.97x inference speedup over the PyTorch
baseline with zero transcription change, and a verified end to end
serving path behind Triton.

## Notes and Limitations

The 2.97x speedup and the argmax agreement were measured on synthetic
input under an isolated benchmark harness on a laptop GPU; production
throughput depends on batching, concurrency, and serving configuration.
The Triton serving benchmark measures sequential HTTP requests at batch
size one; the dynamic batching configuration is provided but not swept
under load in this verification. The word error rate of 1.0000 is
expected for the synthetic signal and is not a model quality metric.
Real speech evaluation requires an ungated Hindi speech corpus; see the
Accuracy section for the procedure. The TensorRT FP16 build emits a
layer norm precision warning from the parser, a known characteristic of
FP16 attention graphs, and its effect is quantified by the numerical
validation rather than ignored.

## License and Attribution

The model `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` is a public
Hugging Face artifact trained within the AI4Bharat effort and released
under its own license terms; verify the model card before commercial
use. This repository contains the optimization and serving pipeline
code and carries no model weights.

  tritonserver --model-repository=/models
python triton\client_infer.py
python triton\bench_serving.py
```

The ONNX artifacts are placed under `models/onnx_model/` and the Triton
model repository is `triton/model_repository/wav2vec2_hindi/1/model.onnx`,
a copy of the optimized graph. After regenerating models, copy the
optimized artifact into the repository directory before starting Triton.

On Linux the same commands apply with forward slashes. The DLL bootstrap
modules are Windows specific; on Linux the CUDA and TensorRT providers
resolve through the standard dynamic loader and the bootstrap calls
require no special handling, but scripts were executed and verified on
Windows and the numbers in this document are the Windows numbers.

Ubuntu with Docker and the NVIDIA Container Toolkit, the server container,
and the model analyzer configuration for batch size and concurrency
sweeping. Production deployment targets Linux because Triton Server is
distributed as a Linux container; development, validation, and the
serving verification above were completed on Windows.


engine profile configuration. The repository therefore maintains two
intentionally distinct artifacts: one optimized for ONNX Runtime and Triton
serving, and one for TensorRT execution.
