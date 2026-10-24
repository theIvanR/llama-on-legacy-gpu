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

**Preliminary result:** On a four-GPU Tesla K40c system, enabling the relevant CUDA graph changes increased observed generation throughput from approximately 35 to over 55 tokens/s in the tested workload.

Results are workload- and configuration-dependent and should be treated as preliminary rather than universal performance guarantees.

### 3. Legacy NCCL Compatibility on Windows

Adds an experimental path for multi-GPU communication using a legacy Windows build of NCCL.

The implementation addresses API differences and missing functionality in the legacy NCCL version, including unavailable grouped-collective APIs and BF16 datatype support.

- Supports an FP32 AllReduce path for the tested legacy NCCL configuration.
- Avoids dependencies on unavailable NCCL APIs.
- Corrects compatibility with the legacy `ncclCommDestroy` signature.
- Enables experimentation with NCCL-based multi-GPU tensor-parallel inference on older hardware.

**Important:** This path is experimental. Correct collective ordering, synchronization, graph-capture behavior, and multi-GPU correctness must be validated against the specific NCCL build.

Initial testing showed an approximately 5-10% end-to-end improvement over the comparison configuration. The result is preliminary and may vary by workload achieving up to approx 60tps.

### 4. Multi-GPU Inference on Older Hardware

The patches are intended to make older multi-GPU systems more practical for local inference by combining improved compute compatibility, reduced launch overhead, and an alternative communication path.

The primary test configuration consists of four Tesla K40c GPUs running a modern mixture-of-experts model.

## Preliminary Performance Results

| Configuration | Observed result |
|---|---:|
| Four Tesla K40c GPUs | 4-GPU inference |
| Model | Unsloth MTP Qwen3.6-35B-A3B, Q8_0 |
| CUDA Toolkit | 11.8 |
| Generation throughput after changes | approx 35 tps -> approx 60ish PG |

These figures describe a specific test system and workload. They are not directly comparable to results obtained with different models, quantizations, context lengths, drivers, or GPU configurations.

A more systematic benchmark suite covering prompt processing, generation, memory usage, and multi-GPU scaling is a work in progress.

## Build Configuration

### Prerequisites

- **CUDA Toolkit:** 11.8
- **Compiler:** Visual Studio 2019 C++ Build Tools (Windows)
- **GPU architecture:** Select the compute capabilities you intend to target.
- **NCCL:** Optional; required only for the experimental NCCL-enabled multi-GPU build.

### 1. Clone the Dependencies

Obtain the following repositories and resources:

- **llama.cpp legacy GPU compatibility changes:** [PR #30727](https://github.com/ggml-org/llama.cpp/pull/30727)  
  Contains the CUDA compatibility and performance changes. Use the PR branch or a checkout containing the relevant patches.

- **NCCL for Windows:** [theIvanR/nccl-on-windows](https://github.com/theIvanR/nccl-on-windows)  
  Provides the legacy Windows NCCL build used by the experimental multi-GPU communication path.

- **Windows build scripts:** [lmstudio-unlocked-backend — Generate Backends/Windows](https://github.com/theIvanR/lmstudio-unlocked-backend/tree/main/Generate%20Backends/Windows)  
  Provides example scripts for configuring and building the CUDA backend.

- **Optional — Tesla K40c BIOS unlocker:** [kepler-on-steroids](https://github.com/theIvanR/kepler-on-steroids/)  
  Relevant if you are using Tesla K40c GPUs and want to experiment with the available BIOS modifications. This is not required to build llama.cpp.

### 2. Configure and Build llama.cpp

1. Open the CUDA/NCCL build script you intend to use, for example `build_gpu_cuda_nccl.cmd`.
2. **Configure the NCCL paths.** Point the script to your `nccl_all_arch` directory, ensuring that the NCCL header, import library, and DLL paths match the files in your local checkout.
3. **Select the target GPU architectures.** Edit the architecture list in the build script to match your hardware. For example, use compute capability `35` for the Tesla K40c (Kepler). Add other architectures as needed.
4. Check that the CUDA Toolkit and Visual Studio toolchain paths are correct.
5. Run the build script from a command prompt:

   ```bat
   build_gpu_cuda_nccl.cmd
   ```

   Adjust the filename if you are using a different builder.

Once compilation completes, locate `llama-server.exe` in the configured build output directory.

**Note:** The NCCL-enabled build is experimental. If you do not need NCCL, you can use a non-NCCL CUDA build instead, provided your selected builder supports it.

### 3. Run Inference

The following is an example Windows batch script for launching a model with `llama-server`. Adjust the executable path, model directory, model filenames, context length, and inference options to suit your system.

```bat
@echo off
setlocal

REM Enable CUDA peer-to-peer support where available
set "GGML_CUDA_P2P=1"

REM Path to the compiled llama.cpp server
set "LLAMA_EXE=C:\Users\Admin\source\llama.cpp\build_gpu_cuda\bin\llama-server.exe"

REM Directory containing the model and multimodal projector
set "MODEL_DIR=C:\Users\Admin\Documents\LLM Models\unsloth\Qwen3.6-35B-A3B-MTP-GGUF"

set "MODEL=Qwen3.6-35B-A3B-Q8_0.gguf"
set "MMPROJ=mmproj-F16.gguf"

pushd "%MODEL_DIR%" || exit /b 1

"%LLAMA_EXE%" ^
  -m "%MODEL%" ^
  --mmproj "%MMPROJ%" ^
  --port 8080 ^
  -ngl 999 ^
  -c 65536 ^
  --fit off ^
  --load-mode none ^
  --split-mode tensor ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 2

set "ERR=%ERRORLEVEL%"

popd

REM exit /b %ERR%

echo.
echo llama-server exit code: %ERR%
echo.
pause
exit /b %ERR%
```

#### Example configuration notes

- `GGML_CUDA_P2P=1` enables the peer-to-peer setting used by this configuration; actual P2P availability depends on the GPUs, driver, and platform.
- `-ngl 999` requests offloading as many model layers as possible to the GPUs.
- `-c 65536` sets a 65,536-token context. Memory requirements depend on the model and cache configuration.
- `--split-mode tensor` selects tensor-based splitting across GPUs.
- `--spec-type draft-mtp --spec-draft-n-max 2` enables the example's MTP speculative decoding configuration. The model and build must support the relevant functionality.
- `--mmproj` supplies the multimodal projector. Omit it if you are running a model that does not require one, and check the options supported by your particular checkout.

**Important:** This is an example launch configuration, not a universal command. Some options may differ between llama.cpp revisions or custom patches. Check `llama-server.exe --help` if an option is rejected.

### 4. Verify the Build

After launching the server:

1. Check the console output for CUDA initialization and GPU offloading.
2. Confirm that all intended GPUs are detected and used.
3. Verify that the model loads successfully and produces valid output.
4. For NCCL builds, test multi-GPU operation and numerical correctness before relying on performance results.

If the build fails, first check the selected compute capabilities, CUDA and compiler paths, and NCCL library locations.
