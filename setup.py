#!/usr/bin/env python3
"""
CrediFi Setup Script
This script helps you set up the CrediFi platform.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print the CrediFi banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🏦 CrediFi Platform 🏦                    ║
    ║                                                              ║
    ║         AI-Powered Decentralized Lending Platform           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "data/processed",
        "models",
        "blockchain/contracts/compiled",
        "ui/pages",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def generate_sample_data():
    """Generate sample credit data."""
    print("\n📊 Generating sample data...")
    
    try:
        # Add project root to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from utils.helpers import create_sample_credit_data
        df = create_sample_credit_data()
        df.to_csv("data/credit_data.csv", index=False)
        
        print(f"✅ Generated sample data with {len(df)} records")
        return True
    except Exception as e:
        print(f"❌ Failed to generate sample data: {e}")
        return False

def test_system():
    """Test the system components."""
    print("\n🧪 Testing system components...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System test passed")
            return True
        else:
            print("❌ System test failed")
            print("Error output:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run system test: {e}")
        return False

def setup_ganache():
    """Provide instructions for setting up Ganache."""
    print("\n🔗 Blockchain Setup Instructions:")
    print("=" * 50)
    print("To use the blockchain features, you need to set up Ganache:")
    print()
    print("1. Install Node.js (if not already installed):")
    print("   https://nodejs.org/")
    print()
    print("2. Install Ganache CLI:")
    print("   npm install -g ganache-cli")
    print()
    print("3. Start Ganache (in a separate terminal):")
    print("   ganache --port 8545 --accounts 10 --defaultBalanceEther 100")
    print()
    print("4. The CrediFi platform will automatically connect to Ganache")
    print("   when you run the application.")
    print()

def print_next_steps():
    """Print next steps for the user."""
    print("\n🚀 Next Steps:")
    print("=" * 30)
    print("1. Start Ganache (if using blockchain features):")
    print("   ganache --port 8545 --accounts 10 --defaultBalanceEther 100")
    print()
    print("2. Run the CrediFi application:")
    print("   streamlit run app.py")
    print()
    print("3. Open your browser and navigate to:")
    print("   http://localhost:8501")
    print()
    print("4. Initialize the system from the Settings page")
    print()
    print("📚 For more information, see the README.md file")

def main():
    """Main setup function."""
    print_banner()
    
    print("Welcome to CrediFi! This script will help you set up the platform.")
    print()
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup failed: Incompatible Python version")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        return False
    
    # Create directories
    create_directories()
    
    # Generate sample data
    if not generate_sample_data():
        print("⚠️ Warning: Could not generate sample data")
    
    # Test system
    if not test_system():
        print("⚠️ Warning: System test failed")
    
    # Setup instructions
    setup_ganache()
    
    # Next steps
    print_next_steps()
    
    print("\n🎉 Setup completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)
