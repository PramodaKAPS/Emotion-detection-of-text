import subprocess
import sys


def install_requirements():
    """Install required packages with compatible versions"""
    packages = [
        "datasets==2.18.0",
        "huggingface_hub==0.24.2",
        "fsspec==2023.10.0",
        "transformers==4.41.2",
        "imbalanced-learn",
        "pandas",
        "numpy",
        "torch",
        "sentencepiece",
        "protobuf==4.25.4"  # Added for DeBERTa-v3 tokenizer conversion support
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
    
    print(" All required packages installed successfully!")
    print(" Please restart your runtime: Runtime > Restart runtime")


def setup_cache_directory():
    """Setup cache directory for Hugging Face models"""
    import os
    
    cache_dir = "/content/huggingface_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Created cache directory: {cache_dir}")
    else:
        print(f"Using cache directory: {cache_dir}")
        # Clear cache if it exists
        os.system(f"rm -rf {cache_dir}/*")
        print(f"Cleared contents of cache directory: {cache_dir}")
    
    return cache_dir


'''def mount_google_drive():
    """Mount Google Drive for saving models"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print(" Google Drive mounted successfully!")
    except ImportError:
        print(" Not in Google Colab environment - skipping drive mount")'''


if __name__ == "__main__":
    install_requirements()

