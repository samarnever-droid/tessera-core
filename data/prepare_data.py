"""
Script to download and prepare benchmark datasets:
1. tiny-shakespeare (character level, ~1MB)
2. enwik8 (character level, ~100MB)
3. wikitext-2 (word/subword level)
"""

import os
import urllib.request
import zipfile

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ENWIK8_URL = "https://data.deepai.org/enwik8.zip"

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"[OK] File already exists: {dest_path}")
        return
    print(f"Downloading {url} -> {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[OK] Downloaded {dest_path}")

def prepare_tiny_shakespeare():
    dest = os.path.join(DATA_DIR, "tiny_shakespeare.txt")
    download_file(SHAKESPEARE_URL, dest)

def prepare_enwik8():
    zip_dest = os.path.join(DATA_DIR, "enwik8.zip")
    txt_dest = os.path.join(DATA_DIR, "enwik8.txt")
    if os.path.exists(txt_dest):
        print(f"[OK] enwik8 already unpacked at {txt_dest}")
        return
    # Fallback to direct download if needed
    try:
        download_file(ENWIK8_URL, zip_dest)
        with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"[OK] Extracted enwik8 to {DATA_DIR}")
    except Exception as e:
        print(f"Note: enwik8 download failed ({e}). Can be manually provided or retried.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Preparing datasets in", DATA_DIR)
    prepare_tiny_shakespeare()
    # Uncomment when ready to test enwik8
    # prepare_enwik8()
