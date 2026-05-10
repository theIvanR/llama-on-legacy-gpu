LLama CPP on Cuda on Kepler GPUs

## 0: configure environment
- install git
- install visual studio (any) supporting v142, recommended 2022 with backwards tool.
- install driver 472.50
- install cuda 11.8
- install cudnn 8.7.0 (drag and drop into nvidia)


0 Fetch Llama

	```bash
	@echo off
	setlocal enabledelayedexpansion

	rem ==========================================
	rem Fetch or update the latest llama.cpp from GitHub
	rem ==========================================

	rem Set target directory
	set "TARGET_DIR=%USERPROFILE%\source\llama.cpp"

	rem Ensure target directory exists
	if not exist "%TARGET_DIR%" (
		echo [INFO] Directory "%TARGET_DIR%" does not exist. Creating...
		mkdir "%TARGET_DIR%"
	)

	rem Navigate to target directory
	pushd "%TARGET_DIR%" >nul 2>&1 || (
		echo [ERROR] Failed to enter "%TARGET_DIR%".
		exit /b 1
	)

	rem Update existing repository or clone if missing
	if exist ".git" (
		echo [INFO] Git repository detected. Pulling latest changes...
		git reset --hard >nul 2>&1
		git clean -fd >nul 2>&1
		git pull origin main
		if errorlevel 1 (
			echo [ERROR] Failed to update repository.
			popd
			exit /b 1
		)
	) else (
		echo [INFO] No Git repository found. Cloning llama.cpp...
		rem Go up to parent directory to safely clone
		pushd .. >nul
		git clone https://github.com/ggerganov/llama.cpp.git "%TARGET_DIR%"
		if errorlevel 1 (
			echo [ERROR] Failed to clone repository.
			popd
			exit /b 1
		)
		popd
	)

	echo [SUCCESS] Latest llama.cpp fetched/updated successfully.
	popd
	pause
	```

1 Activate Environment and Build Llama
- activat env
	```bash
	call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.29
	```
- build llama
	```bash
	@echo off
	setlocal

	REM =========================
	REM User configuration (optionally pass llama dir as first argument
	REM =========================
	set "DEFAULT_SRC_DIR=%USERPROFILE%\source\llama.cpp"

	if "%~1"=="" (set "SRC_DIR=%DEFAULT_SRC_DIR%") else (set "SRC_DIR=%~1")

	pushd "%SRC_DIR%" >nul 2>&1
	if errorlevel 1 (
		echo [ERROR] Invalid source directory: "%SRC_DIR%"
		exit /b 1
	)

	echo [INFO] nvcc version:
	nvcc --version

	REM pause

	REM -------------------------
	REM Set Flags and Clean up
	REM -------------------------
	set "BUILD_DIR=build_gpu_cuda"
	set "CL=/bigobj %CL% /Ot /fp:fast"


	if exist "%BUILD_DIR%" (
		echo [INFO] Removing old build directory "%BUILD_DIR%"...
		
		rmdir /s /q "%BUILD_DIR%"
		
		if errorlevel 1 (
			echo [ERROR] Failed to Remove Dir
			popd
			pause
			exit /b 1
		)
	)


	REM -------------------------
	REM Configure with CMake (set architectures to what you have)
	REM -------------------------
	echo [INFO] Building in "%BUILD_DIR%"
	set "BUILD_FAILED=0"

	cmake -S . -B "%BUILD_DIR%" ^
		  -G "Ninja" ^
		  -DCMAKE_BUILD_TYPE=Release ^
		  -DLLAMA_CURL=OFF ^
		  -DGGML_NATIVE=ON ^
		  -DGGML_CUDA=ON ^
		  -DCMAKE_CUDA_ARCHITECTURES=35
		  
	cmake --build "%BUILD_DIR%" --verbose
	if errorlevel 1 set "BUILD_FAILED=1"

	popd

	if "%BUILD_FAILED%"=="1" (
		echo [ERROR] CMake failed
		pause
		exit /b 1
	)

	echo [SUCCESS] Build finished successfully!
	exit /b 0
	```

2: Run via web UI: multi gpu multimodal example
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
