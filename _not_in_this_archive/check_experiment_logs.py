#!/usr/bin/env python3
"""
Check what happened in the experiment directory
"""
import os
import glob

def check_experiment_directory():
    """Check experiment directory for any logs or errors"""
    
    exp_dir = "/content/MonoX/experiments/ffs_stylegan-v_random_unknown"
    
    print(f"🔍 CHECKING EXPERIMENT DIRECTORY: {exp_dir}")
    
    if not os.path.exists(exp_dir):
        print("❌ Experiment directory does not exist!")
        return
    
    print(f"📁 Directory exists. Contents:")
    
    # List all files recursively
    for root, dirs, files in os.walk(exp_dir):
        level = root.replace(exp_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size} bytes)")
            
            # Check log files
            if file.endswith('.txt') or file.endswith('.log') or 'log' in file.lower():
                print(f"{subindent}📝 Reading log file: {file}")
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if content.strip():
                            print(f"{subindent}📄 Content:")
                            # Show all content for log files
                            for line_num, line in enumerate(content.split('\n'), 1):
                                if line.strip():
                                    print(f"{subindent}   {line_num:3d}: {line}")
                        else:
                            print(f"{subindent}📄 File is empty")
                except Exception as e:
                    print(f"{subindent}❌ Could not read: {e}")
    
    # Also check if there are any recent Python error logs in the system
    print(f"\n🔍 Checking for Python errors...")
    
    # Look for any stderr or error files
    error_patterns = [
        "/tmp/*.log",
        "/tmp/python*.err", 
        f"{exp_dir}/*.err",
        f"{exp_dir}/error*",
        f"{exp_dir}/stderr*"
    ]
    
    for pattern in error_patterns:
        files = glob.glob(pattern)
        for file in files:
            print(f"📝 Found potential error file: {file}")
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if content.strip():
                        print(f"📄 Error content:")
                        print(content[-1000:])  # Last 1000 chars
            except:
                pass

if __name__ == "__main__":
    check_experiment_directory()