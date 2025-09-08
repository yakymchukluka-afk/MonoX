#!/usr/bin/env python3
"""
SSH Key Authentication for Hugging Face Spaces
This script sets up SSH key authentication as an alternative to tokens.
"""

import os
import subprocess
import tempfile
from pathlib import Path
import base64

def setup_ssh_authentication():
    """Setup SSH key authentication for HF Space."""
    print("🔑 Setting up SSH Key Authentication for HF Space")
    print("=" * 50)
    
    # Check if SSH key is provided as secret
    ssh_private_key = os.environ.get('SSH_PRIVATE_KEY')
    if not ssh_private_key:
        print("❌ SSH_PRIVATE_KEY not found in environment")
        print("📝 To use SSH authentication:")
        print("   1. Generate SSH key: ssh-keygen -t ed25519 -C 'your@email.com'")
        print("   2. Add public key to HF account: https://huggingface.co/settings/keys")
        print("   3. Add private key as Space secret: SSH_PRIVATE_KEY")
        print("   4. Restart the Space")
        return False
    
    try:
        # Create .ssh directory
        ssh_dir = Path.home() / '.ssh'
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        
        # Write private key
        private_key_path = ssh_dir / 'id_ed25519'
        with open(private_key_path, 'w') as f:
            f.write(ssh_private_key)
        private_key_path.chmod(0o600)
        
        # Write public key (if provided)
        ssh_public_key = os.environ.get('SSH_PUBLIC_KEY')
        if ssh_public_key:
            public_key_path = ssh_dir / 'id_ed25519.pub'
            with open(public_key_path, 'w') as f:
                f.write(ssh_public_key)
            public_key_path.chmod(0o644)
        
        # Create SSH config
        ssh_config = ssh_dir / 'config'
        with open(ssh_config, 'w') as f:
            f.write("""Host hf.co
    HostName hf.co
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking no
""")
        ssh_config.chmod(0o600)
        
        # Start SSH agent and add key
        subprocess.run(['eval', '$(ssh-agent -s)'], shell=True, check=True)
        subprocess.run(['ssh-add', str(private_key_path)], check=True)
        
        print("✅ SSH key setup completed")
        
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

def setup_git_with_ssh():
    """Configure Git to use SSH for HF repositories."""
    print("\\n🔧 Configuring Git for SSH authentication...")
    
    try:
        # Configure Git to use SSH for HF
        subprocess.run([
            'git', 'config', '--global', 
            'url.git@hf.co:.insteadOf', 'https://huggingface.co/'
        ], check=True)
        
        subprocess.run([
            'git', 'config', '--global', 
            'url.git@hf.co:.insteadOf', 'https://hf.co/'
        ], check=True)
        
        print("✅ Git configured for SSH")
        return True
        
    except Exception as e:
        print(f"❌ Git configuration failed: {e}")
        return False

def create_ssh_upload_function():
    """Create upload function that uses Git with SSH."""
    print("\\n📝 Creating SSH-based upload function...")
    
    upload_code = '''
def upload_via_git_ssh(file_path: str, repo_id: str = "lukua/monox-model") -> bool:
    """Upload file using Git with SSH authentication."""
    import subprocess
    import tempfile
    import shutil
    from pathlib import Path
    
    try:
        # Clone repository using SSH
        repo_url = f"git@hf.co:{repo_id}.git"
        temp_dir = tempfile.mkdtemp()
        repo_dir = Path(temp_dir) / repo_id.split('/')[-1]
        
        # Clone the repository
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
        
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, dest_path)
        
        # Add, commit, and push
        subprocess.run(['git', 'add', str(dest_path.relative_to(repo_dir))], 
                      cwd=repo_dir, check=True)
        
        subprocess.run([
            'git', 'commit', '-m', f'Upload {file_name} via SSH'
        ], cwd=repo_dir, check=True)
        
        subprocess.run(['git', 'push'], cwd=repo_dir, check=True)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print(f"✅ Uploaded via SSH: {file_name}")
        return True
        
    except Exception as e:
        print(f"❌ SSH upload failed: {e}")
        return False
'''
    
    return upload_code

def main():
    """Main function to setup SSH authentication."""
    print("🚀 MonoX SSH Authentication Setup")
    print("=" * 40)
    
    # Setup SSH authentication
    if not setup_ssh_authentication():
        print("\\n❌ SSH setup failed")
        print("\\n📋 Alternative: Use HF_TOKEN instead")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create token with 'Write' permissions")
        print("   3. Add as Space secret: HF_TOKEN")
        return False
    
    # Configure Git
    setup_git_with_ssh()
    
    # Create upload function
    upload_func = create_ssh_upload_function()
    
    print("\\n✅ SSH authentication setup completed!")
    print("\\n📋 Next steps:")
    print("   1. Add SSH_PRIVATE_KEY as Space secret")
    print("   2. Add SSH_PUBLIC_KEY as Space secret (optional)")
    print("   3. Restart the Space")
    print("   4. Use the SSH upload function for file uploads")
    
    return True

if __name__ == "__main__":
    main()