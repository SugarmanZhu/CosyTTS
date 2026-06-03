$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# Add conda env's Library\bin (FFmpeg DLLs) and torch\lib (CUDA + cuDNN DLLs) to PATH
$env:PATH = (Join-Path $PSScriptRoot ".venv\Library\bin") + ";" + `
            (Join-Path $PSScriptRoot ".venv\Lib\site-packages\torch\lib") + ";" + `
            $env:PATH
& .\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8765 --log-level info
