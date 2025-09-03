#!/usr/bin/env python3
"""
HF Clone Issue Diagnostic Script
Tests each potential cause of "Error while cloning repository"
"""

import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, capture_output=True, timeout=30):
    """Run command with timeout and error handling"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, 
            text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)

def test_repository_size():
    """Test 1: Repository size issues"""
    print("🔍 TEST 1: Repository Size Analysis")
    print("=" * 50)
    
    # Total size
    code, stdout, stderr = run_command("du -sh .")
    total_size = stdout.strip().split()[0] if stdout else "Unknown"
    print(f"Total repository size: {total_size}")
    
    # Git directory size
    code, stdout, stderr = run_command("du -sh .git")
    git_size = stdout.strip().split()[0] if stdout else "Unknown"
    print(f"Git directory size: {git_size}")
    
    # Large files
    print("\nLarge files (>1MB):")
    code, stdout, stderr = run_command("find . -size +1M -type f | head -10")
    if stdout:
        print(stdout)
    else:
        print("No large files found")
    
    # File count
    code, stdout, stderr = run_command("find . -type f | wc -l")
    file_count = stdout.strip() if stdout else "Unknown"
    print(f"Total file count: {file_count}")
    
    print(f"✅ Size test complete\n")

def test_submodule_issues():
    """Test 2: Submodule problems"""
    print("🔍 TEST 2: Submodule Issues")
    print("=" * 50)
    
    # Check .gitmodules
    if os.path.exists(".gitmodules"):
        print("📁 .gitmodules exists:")
        with open(".gitmodules", "r") as f:
            print(f.read())
    else:
        print("❌ No .gitmodules file")
        return
    
    # Check submodule status
    print("\n🔍 Submodule status:")
    code, stdout, stderr = run_command("git submodule status")
    print(f"Exit code: {code}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")
    
    # Check if submodule directories exist
    print("\n📂 Submodule directories:")
    if os.path.exists(".external"):
        code, stdout, stderr = run_command("ls -la .external/")
        print(stdout)
    else:
        print("❌ No .external directory")
    
    # Test submodule URL accessibility
    print("\n🌐 Testing submodule URL accessibility:")
    code, stdout, stderr = run_command(
        "curl -s -o /dev/null -w '%{http_code}' https://github.com/yakymchukluka-afk/stylegan-v",
        timeout=10
    )
    print(f"HTTP response: {stdout}")
    
    print(f"✅ Submodule test complete\n")

def test_git_integrity():
    """Test 3: Git repository integrity"""
    print("🔍 TEST 3: Git Repository Integrity")
    print("=" * 50)
    
    # Git fsck
    print("🔧 Git filesystem check:")
    code, stdout, stderr = run_command("git fsck --full", timeout=60)
    print(f"Exit code: {code}")
    if code != 0:
        print(f"❌ Git fsck failed: {stderr}")
    else:
        print("✅ Git repository is clean")
    
    # Check for large objects
    print("\n📦 Large git objects:")
    code, stdout, stderr = run_command(
        "git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -k3 -n | tail -10"
    )
    if stdout:
        print("Top 10 largest objects:")
        print(stdout)
    
    # Branch complexity
    print("\n🌿 Branch analysis:")
    code, stdout, stderr = run_command("git branch -a | wc -l")
    branch_count = stdout.strip() if stdout else "Unknown"
    print(f"Total branches: {branch_count}")
    
    code, stdout, stderr = run_command("git log --oneline | wc -l")
    commit_count = stdout.strip() if stdout else "Unknown"
    print(f"Total commits: {commit_count}")
    
    print(f"✅ Git integrity test complete\n")

def test_clone_simulation():
    """Test 4: Simulate HF clone process"""
    print("🔍 TEST 4: Clone Simulation")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"🧪 Testing clone to: {temp_dir}")
        
        # Test 1: Simple clone (no submodules)
        print("\n1️⃣ Testing simple clone (no submodules):")
        clone_dir = os.path.join(temp_dir, "test_clone_simple")
        code, stdout, stderr = run_command(
            f"git clone --no-recurse-submodules . {clone_dir}",
            timeout=120
        )
        print(f"Exit code: {code}")
        if code != 0:
            print(f"❌ Simple clone failed: {stderr}")
        else:
            print("✅ Simple clone successful")
            # Check size
            code, stdout, stderr = run_command(f"du -sh {clone_dir}")
            print(f"Cloned size: {stdout.strip()}")
        
        # Test 2: Clone with submodules
        print("\n2️⃣ Testing clone with submodules:")
        clone_dir2 = os.path.join(temp_dir, "test_clone_submodules")
        code, stdout, stderr = run_command(
            f"git clone --recurse-submodules . {clone_dir2}",
            timeout=120
        )
        print(f"Exit code: {code}")
        if code != 0:
            print(f"❌ Submodule clone failed: {stderr}")
        else:
            print("✅ Submodule clone successful")
        
        # Test 3: Shallow clone
        print("\n3️⃣ Testing shallow clone:")
        clone_dir3 = os.path.join(temp_dir, "test_clone_shallow")
        code, stdout, stderr = run_command(
            f"git clone --depth 1 . {clone_dir3}",
            timeout=60
        )
        print(f"Exit code: {code}")
        if code != 0:
            print(f"❌ Shallow clone failed: {stderr}")
        else:
            print("✅ Shallow clone successful")
            code, stdout, stderr = run_command(f"du -sh {clone_dir3}")
            print(f"Shallow clone size: {stdout.strip()}")
    
    print(f"✅ Clone simulation complete\n")

def test_docker_context():
    """Test 5: Docker build context issues"""
    print("🔍 TEST 5: Docker Build Context")
    print("=" * 50)
    
    # Test Dockerfile syntax
    print("📋 Dockerfile syntax check:")
    if os.path.exists("Dockerfile"):
        code, stdout, stderr = run_command("docker build --dry-run -f Dockerfile . 2>&1 || echo 'Docker not available'")
        if "Docker not available" in stdout:
            print("⚠️ Docker not available for testing")
        else:
            print(f"Docker syntax check: {code}")
    else:
        print("❌ No Dockerfile found")
    
    # Check .dockerignore
    print("\n📋 Docker ignore file:")
    if os.path.exists(".dockerignore"):
        print("✅ .dockerignore exists")
        with open(".dockerignore", "r") as f:
            print(f.read())
    else:
        print("❌ No .dockerignore file")
    
    # Estimate build context size
    print("\n📦 Build context analysis:")
    code, stdout, stderr = run_command("find . -type f | grep -v '.git/' | wc -l")
    context_files = stdout.strip() if stdout else "Unknown"
    print(f"Files in build context: {context_files}")
    
    print(f"✅ Docker context test complete\n")

def test_file_issues():
    """Test 6: File-specific issues"""
    print("🔍 TEST 6: File Issues")
    print("=" * 50)
    
    # Special characters in filenames
    print("🔤 Files with special characters:")
    code, stdout, stderr = run_command("find . -name '*[^a-zA-Z0-9._/-]*' | head -10")
    if stdout:
        print(stdout)
    else:
        print("✅ No files with special characters")
    
    # Binary files
    print("\n💾 Binary files:")
    code, stdout, stderr = run_command("find . -name '*.bin' -o -name '*.pkl' -o -name '*.pth' -o -name '*.so' | head -10")
    if stdout:
        print(stdout)
    else:
        print("✅ No common binary files found")
    
    # Symlinks
    print("\n🔗 Symbolic links:")
    code, stdout, stderr = run_command("find . -type l")
    if stdout:
        print(stdout)
    else:
        print("✅ No symbolic links")
    
    print(f"✅ File issues test complete\n")

def main():
    """Run all diagnostic tests"""
    print("🚀 HF Clone Issue Diagnostic Script")
    print("=" * 60)
    print("This script will test all potential causes of HF cloning issues\n")
    
    tests = [
        test_repository_size,
        test_submodule_issues,
        test_git_integrity,
        test_clone_simulation,
        test_docker_context,
        test_file_issues
    ]
    
    for i, test in enumerate(tests, 1):
        try:
            test()
        except Exception as e:
            print(f"❌ Test {i} failed with error: {e}\n")
    
    print("🎯 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("Review the results above to identify the root cause.")
    print("Look for ❌ errors and failed tests to pinpoint the issue.")

if __name__ == "__main__":
    main()