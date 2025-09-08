#!/usr/bin/env python3
"""
Complete MonoX Hybrid Authentication Setup
Configures both SSH key and HF token authentication.
"""

import os
import subprocess
import shutil
from pathlib import Path
import time

def setup_environment():
    """Setup the complete environment."""
    print("🚀 Setting up MonoX Hybrid Authentication Environment")
    print("=" * 60)
    
    # Create necessary directories
    directories = [
        "samples",
        "checkpoints", 
        "logs",
        "previews",
        ".huggingface",
        ".ssh"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Set environment variables
    os.environ['HF_HOME'] = '/workspace/.huggingface'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/.huggingface/hub'
    
    print("✅ Environment variables set")
    return True

def setup_ssh_key():
    """Setup SSH key authentication."""
    print("\\n🔑 Setting up SSH Key Authentication...")
    
    # The SSH key fingerprint you provided
    expected_fingerprint = "SHA256:UG7cby7CljmfZn9MJPqsfMy1VfMDzTDBMmZUIJbYDNQ"
    
    try:
        # Create .ssh directory
        ssh_dir = Path('.ssh')
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        
        # Check if SSH key already exists
        private_key_path = ssh_dir / 'id_ed25519'
        if not private_key_path.exists():
            print("⚠️ SSH private key not found, checking common locations...")
            
            # Check common locations for the key
            possible_paths = [
                '/root/.ssh/id_ed25519',
                '/home/user/.ssh/id_ed25519',
                '/workspace/.ssh/id_ed25519',
                '/app/.ssh/id_ed25519'
            ]
            
            key_found = False
            for path in possible_paths:
                if Path(path).exists():
                    shutil.copy2(path, private_key_path)
                    print(f"✅ Found SSH key at {path}")
                    key_found = True
                    break
            
            if not key_found:
                print("❌ SSH private key not found in any location")
                print("📝 The key should be available from previous agent setup")
                return False
        
        # Set proper permissions
        private_key_path.chmod(0o600)
        
        # Configure Git for SSH
        subprocess.run([
            'git', 'config', '--global',
            'url.git@hf.co:.insteadOf', 'https://huggingface.co/'
        ], check=True)
        
        subprocess.run([
            'git', 'config', '--global',
            'url.git@hf.co:.insteadOf', 'https://hf.co/'
        ], check=True)
        
        print("✅ Git configured for SSH")
        
        # Test SSH connection
        result = subprocess.run(
            ['ssh', '-T', 'git@hf.co'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ SSH connection test successful")
            print(f"   Response: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ SSH connection test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ SSH setup failed: {e}")
        return False

def setup_hf_token():
    """Setup HF token authentication."""
    print("\\n🎫 Setting up HF Token Authentication...")
    
    # Get token from HF Space secret
    hf_token = os.environ.get('token')  # HF Space secret name
    
    try:
        if not hf_token:
            print("❌ 'token' secret not found in HF Space")
            print("📝 Add 'token' secret in your HF Space settings")
            print("   Secret name: token")
            print("   Secret value: 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")
            return False
        
        # Save token to file
        token_file = Path('.huggingface/token')
        with open(token_file, 'w') as f:
            f.write(hf_token)
        
        print("✅ HF token configured")
        return True
        
    except Exception as e:
        print(f"❌ Token setup failed: {e}")
        return False

def test_authentication():
    """Test both authentication methods."""
    print("\\n🧪 Testing Authentication Methods...")
    
    try:
        from monox_hybrid_auth import MonoXHybridAuth
        
        auth = MonoXHybridAuth()
        if auth.setup_authentication():
            print(f"✅ Authentication method: {auth.auth_method}")
            
            # Test upload
            if auth.test_upload():
                print("✅ Upload test successful!")
                return True
            else:
                print("❌ Upload test failed")
                return False
        else:
            print("❌ Authentication setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

def create_training_launcher():
    """Create a launcher script for easy training."""
    launcher_content = '''#!/usr/bin/env python3
"""
MonoX Training Launcher
Easy way to start training with hybrid authentication.
"""

import sys
from pathlib import Path

def main():
    print("🚀 Starting MonoX Training with Hybrid Authentication")
    print("=" * 60)
    
    # Check if training script exists
    training_script = Path('monox_training_with_hybrid_auth.py')
    if not training_script.exists():
        print("❌ Training script not found")
        return False
    
    # Import and run training
    try:
        from monox_training_with_hybrid_auth import train_monox_with_hybrid_auth
        return train_monox_with_hybrid_auth()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open('launch_training.py', 'w') as f:
        f.write(launcher_content)
    
    # Make it executable
    os.chmod('launch_training.py', 0o755)
    
    print("✅ Training launcher created")
    return True

def main():
    """Main setup function."""
    print("🎯 MonoX Hybrid Authentication Complete Setup")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    # Setup SSH key
    ssh_success = setup_ssh_key()
    
    # Setup HF token
    token_success = setup_hf_token()
    
    if not ssh_success and not token_success:
        print("❌ No authentication method available")
        return False
    
    # Test authentication
    if not test_authentication():
        print("❌ Authentication test failed")
        return False
    
    # Create launcher
    create_training_launcher()
    
    print("\\n🎉 Setup completed successfully!")
    print("\\n📋 Available commands:")
    print("   python monox_hybrid_auth.py          # Test authentication")
    print("   python monox_training_with_hybrid_auth.py  # Start training")
    print("   python launch_training.py            # Easy launcher")
    
    print("\\n🔗 Repository: https://huggingface.co/lukua/monox-model")
    print("📁 Upload paths:")
    print("   - Samples: samples/")
    print("   - Checkpoints: checkpoints/")
    print("   - Logs: logs/")
    
    return True

if __name__ == "__main__":
    main()