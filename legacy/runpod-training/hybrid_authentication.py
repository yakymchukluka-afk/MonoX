#!/usr/bin/env python3
"""
Hybrid Authentication System for MonoX Training
Supports both SSH keys and HF tokens with automatic fallback.
"""

import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, upload_file, login

class HybridAuthenticator:
    """Handles both SSH and token authentication with fallback."""
    
    def __init__(self):
        self.ssh_available = False
        self.token_available = False
        self.auth_method = None
        
    def setup_authentication(self):
        """Setup authentication with automatic method detection."""
        print("🔐 Setting up Hybrid Authentication")
        print("=" * 40)
        
        # Try SSH first
        if self._setup_ssh():
            self.auth_method = "ssh"
            print("✅ Using SSH key authentication")
            return True
        
        # Fallback to token
        if self._setup_token():
            self.auth_method = "token"
            print("✅ Using HF token authentication")
            return True
        
        print("❌ No authentication method available")
        return False
    
    def _setup_ssh(self):
        """Setup SSH key authentication."""
        ssh_private_key = os.environ.get('SSH_PRIVATE_KEY')
        if not ssh_private_key:
            return False
        
        try:
            # Setup SSH directory
            ssh_dir = Path.home() / '.ssh'
            ssh_dir.mkdir(mode=0o700, exist_ok=True)
            
            # Write private key
            private_key_path = ssh_dir / 'id_ed25519'
            with open(private_key_path, 'w') as f:
                f.write(ssh_private_key)
            private_key_path.chmod(0o600)
            
            # Configure Git for SSH
            subprocess.run([
                'git', 'config', '--global',
                'url.git@hf.co:.insteadOf', 'https://huggingface.co/'
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
                return True
                
        except Exception as e:
            print(f"SSH setup failed: {e}")
        
        return False
    
    def _setup_token(self):
        """Setup HF token authentication."""
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            return False
        
        try:
            login(token=hf_token)
            self.token_available = True
            return True
        except Exception as e:
            print(f"Token setup failed: {e}")
            return False
    
    def upload_file(self, file_path: str, repo_id: str = "lukua/monox-model") -> bool:
        """Upload file using the best available method."""
        if self.auth_method == "ssh":
            return self._upload_via_ssh(file_path, repo_id)
        elif self.auth_method == "token":
            return self._upload_via_token(file_path, repo_id)
        else:
            print("❌ No authentication method available")
            return False
    
    def _upload_via_ssh(self, file_path: str, repo_id: str) -> bool:
        """Upload file using Git with SSH."""
        try:
            import tempfile
            import shutil
            from pathlib import Path
            
            # Clone repository
            repo_url = f"git@hf.co:{repo_id}.git"
            temp_dir = tempfile.mkdtemp()
            repo_dir = Path(temp_dir) / repo_id.split('/')[-1]
            
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
                token=os.environ.get('HF_TOKEN'),
                repo_type="model"
            )
            
            print(f"✅ Uploaded via token: {file_name}")
            return True
            
        except Exception as e:
            print(f"❌ Token upload failed: {e}")
            return False

def main():
    """Test the hybrid authentication system."""
    print("🧪 Testing Hybrid Authentication System")
    print("=" * 40)
    
    auth = HybridAuthenticator()
    
    if auth.setup_authentication():
        print(f"\\n✅ Authentication method: {auth.auth_method}")
        
        # Test upload (create a test file)
        test_file = Path('test_upload.txt')
        with open(test_file, 'w') as f:
            f.write("Test upload from hybrid authentication\\n")
        
        if auth.upload_file(str(test_file)):
            print("✅ Upload test successful!")
        else:
            print("❌ Upload test failed")
        
        # Cleanup
        test_file.unlink()
        
    else:
        print("❌ No authentication method available")
        print("\\n📋 Setup instructions:")
        print("   SSH Method:")
        print("   1. Generate SSH key: ssh-keygen -t ed25519")
        print("   2. Add public key to HF: https://huggingface.co/settings/keys")
        print("   3. Add private key as secret: SSH_PRIVATE_KEY")
        print("\\n   Token Method:")
        print("   1. Create token: https://huggingface.co/settings/tokens")
        print("   2. Add as secret: HF_TOKEN")

if __name__ == "__main__":
    main()