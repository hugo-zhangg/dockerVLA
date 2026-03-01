# ==============================================================================
# [LEARNING LEVEL]: IGNORE (无需关注)
# [ROLE]: 工具脚本。用于自动下载和配置 Libero 数据集。
#         [LOGIC]: 检查目录 -> 配置 ~/.libero/config.yaml -> 从 URL 下载 -> 解压。
#         直接运行即可，不需要阅读代码。
# ==============================================================================

import os
import argparse
import sys
import yaml

# Pre-configure libero to avoid interactive prompt
def setup_libero_config():
    libero_config_dir = os.path.expanduser("~/.libero")
    os.makedirs(libero_config_dir, exist_ok=True)
    config_file = os.path.join(libero_config_dir, "config.yaml")
    
    if not os.path.exists(config_file):
        print(f"Creating default libero config at {config_file}")
        # Default config content that libero expects
        # We point benchmark_root to a default location inside the package or current dir
        # But libero uses this config to know where to look for datasets
        # We can set it to the current working directory or a 'libero' subdir
        
        # Determine a reasonable root. In container, maybe /workspace/libero_data
        # But libero package code uses os.path.dirname(__file__) as default if not specified
        
        # Let's check what keys are expected. Usually just 'benchmark_root' and 'assets_root'
        config_data = {
            "benchmark_root": os.path.abspath("./libero_data"),
            "assets_root": os.path.abspath("./libero_data/assets")
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

setup_libero_config()

# Make sure we can import from libero
sys.path.append("/opt/conda/lib/python3.10/site-packages")

try:
    from libero.libero.utils.download_utils import download_url, DATASET_LINKS
except ImportError:
    print("Could not import libero.libero.utils.download_utils")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download LIBERO Dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Target directory for dataset")
    parser.add_argument("--benchmark_name", type=str, default="libero_spatial", help="Name of the benchmark to download")
    args = parser.parse_args()

    # Create data directory if not exists
    os.makedirs(args.data_dir, exist_ok=True)
    target_dir = os.path.abspath(args.data_dir)
    
    print(f"Starting download for benchmark: {args.benchmark_name}")
    print(f"Target directory: {target_dir}")

    # Manual download logic based on inspected file
    if args.benchmark_name in DATASET_LINKS:
        url = DATASET_LINKS[args.benchmark_name]
        print(f"Downloading from: {url}")
        try:
            # Use wget for visual progress
            import subprocess
            import zipfile
            
            zip_filename = f"{args.benchmark_name}.zip"
            zip_path = os.path.join(target_dir, zip_filename)
            
            print(f"Downloading using wget to {zip_path}...")
            # -c for continue
            subprocess.run(["wget", "-c", url, "-O", zip_path], check=True)
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Optional: remove zip
            # os.remove(zip_path)
            
            print("Download and extraction completed.")
            
            # Verify
            dataset_path = os.path.join(target_dir, args.benchmark_name)
            if os.path.exists(dataset_path):
                 print(f"Verified dataset at: {dataset_path}")
                 files = os.listdir(dataset_path)
                 print(f"Found {len(files)} files.")
            else:
                 print(f"Warning: Expected folder {dataset_path} not found. Content might be in root of data dir.")
                 print(f"Files in data_dir: {os.listdir(target_dir)}")
                 
        except Exception as e:
            print(f"Download failed: {e}")
    else:
        print(f"Error: Benchmark '{args.benchmark_name}' not found in DATASET_LINKS.")
        print(f"Available: {list(DATASET_LINKS.keys())}")

if __name__ == "__main__":
    main()
