# ==============================================================================
# [LEARNING LEVEL]: IGNORE (无需关注)
# [ROLE]: 依赖安装脚本。
#         [LOGIC]: 清理环境变量 -> 调用 pip 安装 TensorFlow CPU 版。
#         仅用于 Docker 容器初始化。
# ==============================================================================
# [TOC]:
# - [IGNORE] Environment Cleanup (Lines 15-20): Clear proxy vars to avoid pip errors.
# - [IGNORE] Pip Install (Lines 23-31): Install tensorflow-cpu from Tuna mirror.
# ==============================================================================

import os
import subprocess
import sys

# 彻底清除所有可能的代理变量
keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
for k in keys:
    if k in os.environ:
        del os.environ[k]

print("Environment cleaned. Installing packages...")

# 使用 subprocess 调用 pip，确保环境继承
cmd = [
    sys.executable, "-m", "pip", "install", 
    "tensorflow-cpu", "tensorflow_datasets",
    "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
    "--trust-host", "pypi.tuna.tsinghua.edu.cn"
]

subprocess.check_call(cmd)
print("Done!")
