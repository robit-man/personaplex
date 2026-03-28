#!/usr/bin/env python3
import os
import requests
import json

MODEL_REPO = "cudabenchmarktest/personaplex-7b-turbo2bit"
LOCAL_DIR = "models/personaplex-7b-turbo2bit"

# Get files from HuggingFace API
files_url = f"https://huggingface.co/api/models/{MODEL_REPO}/tree/main"
print(f"Fetching files from {files_url}")
response = requests.get(files_url)
files = response.json()

# Filter for important files (safetensors, config, tokenizer)
important_files = []
for f in files:
    if isinstance(f, dict) and f.get('type') == 'file':
        path = f.get('path', '')
        if '.safetensors' in path or 'config.json' in path or 'tokenizer' in path or 'generation_config' in path:
            important_files.append(f)

print(f"Found {len(important_files)} important files:")
for f in important_files:
    size = f.get('lfs', {}).get('size', f.get('size', 0))
    print(f"  - {f.get('path')} ({size / 1024 / 1024:.2f} MB)")

# Download files
os.makedirs(LOCAL_DIR, exist_ok=True)

for f in important_files:
    file_path = f.get('path', '')
    local_path = os.path.join(LOCAL_DIR, file_path)
    
    # Skip if already exists
    if os.path.exists(local_path):
        print(f"Skipping {file_path} (already exists)")
        continue
    
    os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else '.', exist_ok=True)
    
    raw_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{file_path}"
    
    print(f"Downloading {file_path}...")
    response = requests.get(raw_url, stream=True)
    
    with open(local_path, 'wb') as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            out_file.write(chunk)
    
    print(f"  Downloaded: {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.2f} MB)")

print("\nDownload complete!")
print(f"Files saved to: {LOCAL_DIR}")
