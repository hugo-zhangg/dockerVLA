#!/bin/bash
# ==============================================================================
# [LEARNING LEVEL]: SKIM
# [ROLE]: 本地打包脚本。将项目打包发送到云端，自动忽略数据、缓存和日志。
# ==============================================================================

# 获取当前文件夹名 (通常是 dockerVLA)
PROJECT_DIR=${PWD##*/}
ARCHIVE_NAME="dockerVLA_cloud_ready.tar.gz"

echo "Packing project into ${ARCHIVE_NAME}..."

# 退到上一级目录进行打包，这样解压后还是带有 dockerVLA 文件夹结构
cd ..

tar -czvf ${ARCHIVE_NAME} \
    --exclude="${PROJECT_DIR}/data" \
    --exclude="${PROJECT_DIR}/checkpoints" \
    --exclude="${PROJECT_DIR}/wandb" \
    --exclude="${PROJECT_DIR}/runs" \
    --exclude="${PROJECT_DIR}/__pycache__" \
    --exclude="${PROJECT_DIR}/.git" \
    --exclude="${PROJECT_DIR}/.vscode" \
    --exclude="${PROJECT_DIR}/*.log" \
    --exclude="${PROJECT_DIR}/*.txt" \
    --exclude="${PROJECT_DIR}/*.jpg" \
    --exclude="${PROJECT_DIR}/agent-tools" \
    ${PROJECT_DIR}

# 移回原目录
mv ${ARCHIVE_NAME} ${PROJECT_DIR}/
cd ${PROJECT_DIR}

echo ""
echo "✅ Packing complete! File: ${ARCHIVE_NAME}"
echo "   Size: $(du -h ${ARCHIVE_NAME} | cut -f1)"
echo "--------------------------------------------------"
echo "Next step: SCP this file to your cloud server."
echo "Example: scp ${ARCHIVE_NAME} user@cloud_ip:~/workspace/"
echo "--------------------------------------------------"
