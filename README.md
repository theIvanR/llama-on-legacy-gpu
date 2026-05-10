# LLama CPP on cuda on legacy systems, kepler k40c focused

## 0: configure environment
- install git
- install visual studio (any) supporting v142, recommended 2022 with backwards tool.
- install driver 472.50
- install cuda 11.8
- install cudnn 8.7.0 (drag and drop into nvidia)


## 1 Fetch Llama
- run "fetch_llama.cmd" from https://github.com/theIvanR/lmstudio-unlocked-backend/tree/main/Generate%20Backends/Windows

## 2 Activate Environment and Build Llama
- activat env
	```bash
	call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.29
	```
- download builder "build_gpu_cuda.cmd" from https://github.com/theIvanR/lmstudio-unlocked-backend/tree/main/Generate%20Backends/Windows
- set desired gpu architectures (if multiple gpus) and run
  
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
	  --tensor-split 0.25,0.25,0.25,0.25 ^
	  --port 8080

	set "ERR=%ERRORLEVEL%"
	popd
	exit /b %ERR%
	```
