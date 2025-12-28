@echo off

echo [1/4] Creating Virtual Environment for CUDA...
python -m venv venv_cuda

echo [2/4] Activating Environment and Installing Base Libraries...
call venv_cuda\Scripts\activate

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
echo [3/4] Downloading microsoft/Phi-3.5-mini-instruct-onnx (gpu variant)...
hf download microsoft/Phi-3.5-mini-instruct-onnx --include gpu/* --local-dir .

echo.
echo [4/4] Installing onnxruntime CUDA...
pip install onnxruntime-genai-cuda

echo.
echo SETUP COMPLETE! To run the CUDA version, use:
echo venv_cuda\Scripts\activate
echo python main.py -m gpu/gpu-int4-awq-block-128 -e cuda
pause