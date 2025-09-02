#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE PARAMETER DETECTION SCRIPT
============================================
This script runs the training command with HYDRA_FULL_ERROR=1 to capture
ALL missing configuration parameters at once, then generates the complete
fix automatically.
"""

import subprocess
import re
import sys
import os

def run_training_and_capture_errors():
    """Run training and capture all OmegaConf errors to identify missing parameters."""
    print("🔍 COMPREHENSIVE PARAMETER DETECTION")
    print("="*60)
    print("🎯 Running training with full error reporting to detect ALL missing parameters...")
    
    # Set up environment for full error reporting
    env = os.environ.copy()
    env['HYDRA_FULL_ERROR'] = '1'
    env['PYTHONPATH'] = '/workspace/.external/stylegan-v/src'
    
    # Minimal command to trigger parameter validation
    cmd = [
        'python3', '/workspace/train_super_gpu_forced.py',
        'exp_suffix=param_detect',
        'dataset.path=/content/drive/MyDrive/MonoX/dataset',
        'dataset.resolution=256',
        'training.total_kimg=1',
        'num_gpus=1'
    ]
    
    print(f"🚀 Command: {' '.join(cmd)}")
    print("="*60)
    
    # Run multiple iterations to catch different errors
    all_missing_params = set()
    max_iterations = 10
    
    for iteration in range(max_iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
                cwd='/workspace'
            )
            
            # Parse output for ConfigAttributeError
            output = result.stdout + result.stderr
            
            # Look for missing key errors
            missing_key_patterns = [
                r"Key '([^']+)' is not in struct\s+full_key:\s*([^\s]+)",
                r"Missing key\s+([^\s]+)",
                r"ConfigAttributeError.*full_key:\s*([^\s]+)"
            ]
            
            found_new_error = False
            for pattern in missing_key_patterns:
                matches = re.findall(pattern, output, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) >= 2:
                            missing_param = match[1]  # full_key
                        else:
                            missing_param = match[0]
                    else:
                        missing_param = match
                    
                    if missing_param not in all_missing_params:
                        all_missing_params.add(missing_param)
                        found_new_error = True
                        print(f"❌ Found missing parameter: {missing_param}")
            
            if not found_new_error:
                if "training_loop() function called" in output:
                    print("🎉 SUCCESS! Training loop reached!")
                    break
                elif "ConfigAttributeError" not in output and result.returncode == 0:
                    print("✅ No more configuration errors detected!")
                    break
                else:
                    print("⚠️  Unknown error or no new missing parameters found")
                    break
            
        except subprocess.TimeoutExpired:
            print("⏰ Command timed out - this might indicate progress!")
            break
        except Exception as e:
            print(f"❌ Error running command: {e}")
            break
    
    return list(all_missing_params)

def generate_comprehensive_fix(missing_params):
    """Generate a comprehensive fix with all missing parameters."""
    print("\n🔧 GENERATING COMPREHENSIVE FIX")
    print("="*60)
    
    if not missing_params:
        print("✅ No missing parameters detected!")
        return
    
    print(f"📋 Found {len(missing_params)} missing parameters:")
    for param in sorted(missing_params):
        print(f"   - {param}")
    
    # Generate parameter mappings
    param_fixes = []
    
    # Default values for common parameters
    default_values = {
        'training.target': '0.6',
        'training.augpipe': 'bgc',
        'training.freezed': '0',
        'training.dry_run': 'false',
        'training.cond': 'false',
        'training.nhwc': 'false',
        'training.resume': 'null',
        'model.optim.generator.beta1': '0.0',
        'model.optim.generator.beta2': '0.99',
        'model.optim.discriminator.beta1': '0.0',
        'model.optim.discriminator.beta2': '0.99',
        'model.loss_kwargs.pl_weight': '0.0',
        'model.generator.use_noise': 'false',
        'model.generator.c_dim': '${dataset.c_dim}',
        'model.discriminator.c_dim': '${dataset.c_dim}',
    }
    
    # Generate fixes for each missing parameter
    for param in sorted(missing_params):
        if param in default_values:
            value = default_values[param]
        else:
            # Infer reasonable defaults based on parameter name
            if 'lr' in param.lower():
                value = '0.002'
            elif 'weight' in param.lower():
                value = '0.0'
            elif 'prob' in param.lower():
                value = '0.0'
            elif 'dim' in param.lower():
                value = '512'
            elif param.endswith('.resume'):
                value = 'null'
            elif param.endswith('.seed'):
                value = '0'
            elif any(x in param.lower() for x in ['batch', 'size']):
                value = '8'
            elif any(x in param.lower() for x in ['fp32', 'nobench', 'allow_tf32', 'mirror', 'dry_run', 'cond', 'nhwc']):
                value = 'false'
            elif 'workers' in param.lower():
                value = '8'
            elif 'target' in param.lower():
                value = '0.6'
            elif 'aug' in param.lower():
                value = 'ada'
            else:
                value = 'null'
        
        param_fixes.append(f'            f"+{param}={value}",  # ADD {param} (auto-detected)')
    
    print(f"\n📝 Generated fixes for {len(param_fixes)} parameters:")
    print("```python")
    for fix in param_fixes:
        print(fix)
    print("```")
    
    return param_fixes

def apply_comprehensive_fix(param_fixes):
    """Apply all parameter fixes to the training script at once."""
    if not param_fixes:
        return
    
    print("\n🔧 APPLYING COMPREHENSIVE FIX")
    print("="*60)
    
    # Read current training script
    script_path = '/workspace/train_super_gpu_forced.py'
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find the insertion point (before visualizer parameters)
    insertion_pattern = r'(\s+f"\+training\.p=null",.*?\n)(\s+f"visualizer\.save_every_kimg=)'
    
    match = re.search(insertion_pattern, content, re.DOTALL)
    if match:
        # Insert all new parameters
        new_params = '\n'.join(param_fixes) + '\n'
        new_content = content[:match.end(1)] + new_params + match.group(2)
        
        # Write back to file
        with open(script_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Applied {len(param_fixes)} parameter fixes to {script_path}")
        return True
    else:
        print("❌ Could not find insertion point in training script")
        return False

def main():
    """Main execution function."""
    print("🚀 COMPREHENSIVE STYLEGAN-V PARAMETER DETECTION")
    print("="*60)
    print("This script will detect ALL missing parameters at once and fix them!")
    print()
    
    # Step 1: Detect all missing parameters
    missing_params = run_training_and_capture_errors()
    
    # Step 2: Generate comprehensive fix
    param_fixes = generate_comprehensive_fix(missing_params)
    
    # Step 3: Apply fix
    if param_fixes:
        if apply_comprehensive_fix(param_fixes):
            print("\n🎉 COMPREHENSIVE FIX APPLIED!")
            print("="*60)
            print("✅ All detected missing parameters have been added!")
            print("🚀 Run the training script again to test the complete fix!")
        else:
            print("\n❌ FAILED TO APPLY FIX")
            print("="*60)
            print("Please manually apply the generated parameter fixes.")
    else:
        print("\n🎉 NO MISSING PARAMETERS DETECTED!")
        print("="*60)
        print("✅ Configuration appears to be complete!")

if __name__ == "__main__":
    main()