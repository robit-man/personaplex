#!/usr/bin/env python3
"""Download tokenizer and mimi files from public alternative sources"""
import os
import requests

LOCAL_DIR = "models/personaplex-7b-turbo2bit"
os.makedirs(LOCAL_DIR, exist_ok=True)

# Public alternative sources
FILES_TO_DOWNLOAD = [
    {
        "name": "tokenizer_spm_32k_3.model",
        "url": "https://huggingface.co/kyutai/moshika-pytorch-bf16/resolve/main/tokenizer_spm_32k_3.model"
    },
    {
        "name": "tokenizer-e351c8d8-checkpoint125.safetensors",
        "url": "https://huggingface.co/kyutai/moshika-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors"
    },
]

for file_info in FILES_TO_DOWNLOAD:
    file_name = file_info["name"]
    url = file_info["url"]
    local_path = os.path.join(LOCAL_DIR, file_name)
    
    # Skip if already exists and is valid
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
        print(f"Skipping {file_name} (already exists: {os.path.getsize(local_path)} bytes)")
        continue
    
    print(f"Downloading {file_name}...")
    print(f"  URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(local_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  ✓ Downloaded: {local_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\nDownload complete!")
print(f"Files saved to: {LOCAL_DIR}")
