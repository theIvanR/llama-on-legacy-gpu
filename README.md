# High Performance llama.cpp for legacy systems
A performance and compatibility fork of llama.cpp focused on extending CUDA support and improving inference performance on older NVIDIA GPUs, including Kepler, Maxwell, Pascal, and other pre-Ampere architectures.

The goal is simple: keep legacy GPUs useful for modern LLM inference instead of treating them as an afterthought.

## Target Platforms

- **CUDA Toolkit:** 11.8
- **Primary target:** NVIDIA compute capabilities 3.5–6.2
- **Additional architectures:** Support intended to extend through compute capability 9.0, subject to the selected CUDA toolkit, driver, and build configuration.
- **Test hardware:** NVIDIA Tesla K40c (Kepler, CC 3.5), including multi-GPU configurations.
- **Operating system:** Windows support, including experimental legacy NCCL integration.

> Architecture support depends on the individual kernel, CUDA toolkit, driver, and enabled build options. Successful compilation does not guarantee that every feature is supported on every GPU.

## Patch Overview

### 1. Legacy GPU cuBLAS Compatibility

Improves compatibility with older GPUs by making cuBLAS algorithm and computation-type selection more robust.

- Adds more appropriate computation-type selection for legacy architectures.
- Provides FP32 fallbacks where newer reduced-precision paths are unavailable.
- Avoids selecting unsupported TF32 execution paths on architectures that do not support them.
- Adds diagnostic logging to help identify the selected computation type.

**Objective:** Make matrix multiplication work reliably on older hardware without unnecessarily depending on newer GPU features.

### 2. CUDA Graphs on Legacy Architectures

Enables and tests CUDA graph execution on older GPUs where supported by the underlying runtime and operations.

CUDA graphs can reduce CPU-side launch overhead by capturing and replaying sequences of GPU operations.

- Extends graph execution to legacy GPU configurations.
- Reduces repeated kernel-launch overhead in suitable workloads.
- Can improve token-generation throughput, particularly in multi-GPU inference configurations.

**Preliminary result:** On a four-GPU Tesla K40c system, enabling the relevant CUDA graph changes increased observed generation throughput from approximately 45 to over 55 tokens/s in the tested workload.

Results are workload- and configuration-dependent and should be treated as preliminary rather than universal performance guarantees.

### 3. Legacy NCCL Compatibility on Windows

Adds an experimental path for multi-GPU communication using a legacy Windows build of NCCL.

The implementation addresses API differences and missing functionality in the legacy NCCL version, including unavailable grouped-collective APIs and BF16 datatype support.

- Supports an FP32 AllReduce path for the tested legacy NCCL configuration.
- Avoids dependencies on unavailable NCCL APIs.
- Corrects compatibility with the legacy `ncclCommDestroy` signature.
- Enables experimentation with NCCL-based multi-GPU tensor-parallel inference on older hardware.

**Important:** This path is experimental. Correct collective ordering, synchronization, graph-capture behavior, and multi-GPU correctness must be validated against the specific NCCL build.

Initial testing showed an approximately 5% end-to-end improvement over the comparison configuration. The result is preliminary and may vary by workload.

### 4. Multi-GPU Inference on Older Hardware

The patches are intended to make older multi-GPU systems more practical for local inference by combining improved compute compatibility, reduced launch overhead, and an alternative communication path.

The primary test configuration consists of four Tesla K40c GPUs running a modern mixture-of-experts model.

## Preliminary Performance Results

| Configuration | Observed result |
|---|---:|
| Four Tesla K40c GPUs | 4-GPU inference |
| Model | Qwen3.6-35B-A3B, Q8_0 |
| CUDA Toolkit | 11.8 |
| Generation throughput after changes | 30ish tps -> 60ish tps PG |

These figures describe a specific test system and workload. They are not directly comparable to results obtained with different models, quantizations, context lengths, drivers, or GPU configurations.

A more systematic benchmark suite covering prompt processing, generation, memory usage, and multi-GPU scaling is a work in progress.

## Build Configuration

The current development configuration uses:

- CUDA Toolkit 11.8
- Visual Studio 2019 C++ build tools on Windows
- Legacy NVIDIA GPUs with supported compute capabilities
- An optional experimental NCCL build for multi-GPU communication


## Quickstart
- run: git clone https://github.com/ggml-org/llama.cpp/pull/30727

## 1: Download builder of your choice from 
- download builder "build_gpu_cuda.cmd" from https://github.com/theIvanR/lmstudio-unlocked-backend/tree/main/Generate%20Backends/Windows
  
## 3: Enjoy via WEB API
- go to your build directory and run
- download your models and set which to use. Open in browser. 
	```bash
	@echo off
	setlocal

	set "LLAMA_EXE=C:\Users\Admin\source\llama.cpp\build_gpu_cuda\bin\llama-server.exe"
	set "MODEL_DIR=C:\Users\Admin\Documents\LLM Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF"
	set "MODEL=Qwen3.6-35B-A3B-Q4_K_M.gguf"
	set "MMPROJ=mmproj-Qwen3.6-35B-A3B-BF16.gguf"

	pushd "%MODEL_DIR%" || exit /b 1

	"%LLAMA_EXE%" ^
	  -m "%MODEL%" ^
	  --mmproj "%MMPROJ%" ^
	  --no-mmproj-offload
	  -ngl 999 ^
	  --port 8080

	set "ERR=%ERRORLEVEL%"
	popd
	exit /b %ERR%
	```
