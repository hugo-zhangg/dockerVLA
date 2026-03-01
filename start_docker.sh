#!/bin/bash
echo "Starting Docker container for VLA Project..."

# Ensure image exists or user can build with:
# docker build -t vla_project:latest .

# Run the container
# --gpus all: Enable GPU access inside container (use --gpus '"device=6,7"' for specific GPUs)
# --ipc=host: Share host IPC namespace (crucial for multi-process DataLoader)
# -v $(pwd):/workspace: Mount current directory to /workspace
# -it: Interactive terminal
# --rm: Remove container after exit

docker run --gpus all \
    -it \
    --rm \
    --name vla_dev_env \
    --ipc=host \
    -v "$(pwd)":/workspace \
    vla_project:latest

echo "Container stopped."
