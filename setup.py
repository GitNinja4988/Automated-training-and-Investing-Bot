import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Additional setup for Windows
    if platform.system() == "Windows":
        print("Performing additional Windows setup...")
        # Check for CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                print("CUDA is available. GPU acceleration will be used.")
            else:
                print("CUDA is not available. Using CPU only.")
        except ImportError:
            print("PyTorch installation failed. Please try installing it manually:")
            print("pip install torch torchvision torchaudio")

def create_env_file():
    """Create .env file template if it doesn't exist"""
    if not os.path.exists(".env"):
        print("Creating .env file template...")
        with open(".env", "w") as f:
            f.write("""# API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
""")
        print("Please edit the .env file with your API keys")

def main():
    print("Setting up Automated Trading Bot...")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file template
    create_env_file()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file with your API keys")
    print("2. Run the bot using: python trading_bot.py")
    print("\nNote: If you encounter any issues, please ensure you have:")
    print("- Latest Windows updates installed")
    print("- Python 3.8 or higher installed")
    print("- Sufficient disk space for model training")

if __name__ == "__main__":
    main() 