@echo off

echo [1/4] Creating Virtual Environment for (Windows) CPU...
python -m venv venv_cpu

echo [2/4] Activating Environment and Installing Base Libraries...
call venv_cpu\Scripts\activate

echo.
echo Please enter the pip command to install PyTorch for your venv:
echo (Example: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128)
echo (Press Enter to skip)
set /p "TORCH_CMD= Enter: "

echo.
echo Installing PyTorch...
%TORCH_CMD%

echo.
pip install -r requirements.txt

echo.
echo [3/4] Downloading microsoft/Phi-3.5-mini-instruct-onnx (cpu variant)...
hf download microsoft/Phi-3.5-mini-instruct-onnx --include cpu_and_mobile/* --local-dir .

echo.
echo [4/4] Installing onnxruntime...
pip install onnxruntime-genai

echo.
echo SETUP COMPLETE! To run the CPU version, use:
echo venv_cpu\Scripts\activate
echo python main.py -m cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4 -e cpu
pause
