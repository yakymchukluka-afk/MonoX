#!/usr/bin/env python3
"""
MonoX Hybrid Authentication System
Uses both SSH key and HF token with automatic fallback.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import HfApi, upload_file, login
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonoXHybridAuth:
    """Hybrid authentication system for MonoX training."""
    
    def __init__(self):
        self.ssh_available = False
        self.token_available = False
        self.auth_method = None
        self.hf_token = os.environ.get('HF_TOKEN')
        self.ssh_key_fingerprint = "SHA256:UG7cby7CljmfZn9MJPqsfMy1VfMDzTDBMmZUIJbYDNQ"
        
    def setup_authentication(self):
        """Setup authentication with both SSH and token."""
        print("🔐 Setting up MonoX Hybrid Authentication")
        print("=" * 50)
        
        # Set the token in environment
        os.environ['HF_TOKEN'] = self.hf_token
        
        # Try both methods
        ssh_success = self._setup_ssh()
        token_success = self._setup_token()
        
        if ssh_success and token_success:
            self.auth_method = "hybrid"
            print("✅ Both SSH and token authentication available")
            print("   Using SSH for Git operations, token for API operations")
        elif ssh_success:
            self.auth_method = "ssh"
            print("✅ SSH authentication available")
        elif token_success:
            self.auth_method = "token"
            print("✅ Token authentication available")
        else:
            print("❌ No authentication method available")
            return False
        
        return True
    
    def _setup_ssh(self):
        """Setup SSH key authentication."""
        print("\\n🔑 Setting up SSH authentication...")
        
        try:
            # Check if SSH key is already available
            ssh_dir = Path.home() / '.ssh'
            ssh_dir.mkdir(mode=0o700, exist_ok=True)
            
            # Check for existing SSH key
            private_key_path = ssh_dir / 'id_ed25519'
            if not private_key_path.exists():
                print("⚠️ SSH private key not found, trying to locate...")
                # Try to find the key in common locations
                possible_paths = [
                    '/root/.ssh/id_ed25519',
                    '/home/user/.ssh/id_ed25519',
                    '/workspace/.ssh/id_ed25519'
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        shutil.copy2(path, private_key_path)
                        print(f"✅ Found SSH key at {path}")
                        break
                else:
                    print("❌ SSH private key not found in any location")
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
            
            # Test SSH connection
            result = subprocess.run(
                ['ssh', '-T', 'git@hf.co'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.ssh_available = True
                print("✅ SSH connection test successful")
                print(f"   Response: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ SSH connection test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ SSH setup failed: {e}")
            return False
    
    def _setup_token(self):
        """Setup HF token authentication."""
        print("\\n🎫 Setting up token authentication...")
        
        try:
            login(token=self.hf_token)
            self.token_available = True
            print("✅ Token authentication successful")
            return True
        except Exception as e:
            print(f"❌ Token setup failed: {e}")
            return False
    
    def upload_file(self, file_path: str, repo_id: str = "lukua/monox-model") -> bool:
        """Upload file using the best available method."""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        # Try token first (more reliable for API operations)
        if self.token_available:
            if self._upload_via_token(file_path, repo_id):
                return True
        
        # Fallback to SSH if token fails
        if self.ssh_available:
            if self._upload_via_ssh(file_path, repo_id):
                return True
        
        print("❌ All upload methods failed")
        return False
    
    def _upload_via_token(self, file_path: str, repo_id: str) -> bool:
        """Upload file using HF Hub API with token."""
        try:
            # Determine upload path
            file_name = Path(file_path).name
            if file_path.endswith(('.pkl', '.pt', '.pth', '.ckpt')):
                repo_path = f"checkpoints/{file_name}"
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                repo_path = f"samples/{file_name}"
            elif file_path.endswith('.log'):
                repo_path = f"logs/{file_name}"
            else:
                repo_path = file_name
            
            # Upload with token
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=self.hf_token,
                repo_type="model"
            )
            
            print(f"✅ Uploaded via token: {file_name} -> {repo_path}")
            return True
            
        except Exception as e:
            print(f"❌ Token upload failed: {e}")
            return False
    
    def _upload_via_ssh(self, file_path: str, repo_id: str) -> bool:
        """Upload file using Git with SSH."""
        try:
            # Clone repository
            repo_url = f"git@hf.co:{repo_id}.git"
            temp_dir = tempfile.mkdtemp()
            repo_dir = Path(temp_dir) / repo_id.split('/')[-1]
            
            print(f"📥 Cloning repository: {repo_url}")
            subprocess.run([
                'git', 'clone', repo_url, str(repo_dir)
            ], check=True, cwd=temp_dir)
            
            # Determine destination path
            file_name = Path(file_path).name
            if file_path.endswith(('.pkl', '.pt', '.pth', '.ckpt')):
                dest_path = repo_dir / 'checkpoints' / file_name
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                dest_path = repo_dir / 'samples' / file_name
            elif file_path.endswith('.log'):
                dest_path = repo_dir / 'logs' / file_name
            else:
                dest_path = repo_dir / file_name
            
            # Create destination directory and copy file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
            
            # Git operations
            subprocess.run(['git', 'add', str(dest_path.relative_to(repo_dir))], 
                          cwd=repo_dir, check=True)
            subprocess.run(['git', 'commit', '-m', f'Upload {file_name} via SSH'], 
                          cwd=repo_dir, check=True)
            subprocess.run(['git', 'push'], cwd=repo_dir, check=True)
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            print(f"✅ Uploaded via SSH: {file_name}")
            return True
            
        except Exception as e:
            print(f"❌ SSH upload failed: {e}")
            return False
    
    def test_upload(self):
        """Test the upload functionality."""
        print("\\n🧪 Testing upload functionality...")
        
        # Create test file
        test_file = Path('test_hybrid_upload.txt')
        with open(test_file, 'w') as f:
            f.write("MonoX Hybrid Authentication Test\\n")
            f.write(f"Timestamp: {time.time()}\\n")
            f.write(f"Auth method: {self.auth_method}\\n")
        
        # Test upload
        success = self.upload_file(str(test_file))
        
        # Cleanup
        test_file.unlink()
        
        return success

def main():
    """Main function to setup and test hybrid authentication."""
    print("🚀 MonoX Hybrid Authentication Setup")
    print("=" * 40)
    
    auth = MonoXHybridAuth()
    
    if auth.setup_authentication():
        print(f"\\n✅ Authentication method: {auth.auth_method}")
        
        # Test upload
        if auth.test_upload():
            print("\\n🎉 Hybrid authentication working perfectly!")
            print("\\n📋 Ready for MonoX training with automatic uploads")
        else:
            print("\\n❌ Upload test failed")
    else:
        print("\\n❌ Authentication setup failed")

if __name__ == "__main__":
    import time
    main()