@echo off
echo Starting Docker container for VLA Project...

REM Build the image if it doesn't exist (optional check, or user can run docker build manually)
REM docker build -t vla_project:latest .

REM Run the container
REM --gpus all: Enable GPU access inside container (use --gpus '"device=6,7"' for specific GPUs)
REM --ipc=host: Share host IPC namespace (crucial for multi-process DataLoader)
REM -v %cd%:/workspace: Mount current directory to /workspace
REM -it: Interactive terminal
REM --rm: Remove container after exit

REM docker run --gpus all ^
REM     -it ^

docker run ^
    -it ^
    --rm ^
    --name vla_dev_env ^
    --ipc=host ^
    -v "%cd%":/workspace ^
    vla_project:latest

echo Container stopped.
