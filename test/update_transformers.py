import subprocess
import sys

def update_transformers():
    print("Updating transformers library to latest version...")
    
    # Try the standard upgrade first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
        print("Successfully upgraded transformers library.")
        return True
    except subprocess.CalledProcessError:
        print("Standard upgrade failed. Trying to install from GitHub...")
    
    # If standard upgrade fails, try installing from GitHub
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/huggingface/transformers.git"])
        print("Successfully installed transformers from GitHub.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to update transformers: {e}")
        return False

if __name__ == "__main__":
    success = update_transformers()
    if success:
        print("\nNow you can run your models with proper PhoBERT support.")
    else:
        print("\nFailed to update the transformers library. Please try manually with:")
        print("pip install --upgrade transformers")
        print("or")
        print("pip install git+https://github.com/huggingface/transformers.git")
