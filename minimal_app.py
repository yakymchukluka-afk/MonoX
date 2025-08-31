#!/usr/bin/env python3
"""
Minimal MonoX Training Interface that works without external dependencies.
This version provides basic functionality and clear error messages.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import optional dependencies
HAS_GRADIO = False
HAS_TORCH = False
HAS_FASTAPI = False

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import fastapi
    HAS_FASTAPI = True
except ImportError:
    pass

class SimpleTrainingManager:
    """Simple training manager that works without external dependencies."""
    
    def __init__(self):
        self.setup_workspace()
        self.training_status = "idle"
        self.training_logs = []
    
    def setup_workspace(self):
        """Setup basic workspace structure."""
        directories = ["logs", "checkpoints", "previews", "dataset"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "timestamp": time.time(),
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "has_gradio": HAS_GRADIO,
            "has_torch": HAS_TORCH,
            "has_fastapi": HAS_FASTAPI,
            "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
            "gpu_count": torch.cuda.device_count() if HAS_TORCH else 0,
            "training_status": self.training_status,
            "workspace_structure": {
                "logs": os.path.exists("logs"),
                "checkpoints": os.path.exists("checkpoints"),
                "previews": os.path.exists("previews"),
                "dataset": os.path.exists("dataset"),
                "launch_script": os.path.exists("src/infra/launch.py")
            }
        }
    
    def start_training(self, dataset_path: str = "/workspace/dataset", **kwargs) -> Dict[str, Any]:
        """Attempt to start training."""
        if not os.path.exists("src/infra/launch.py"):
            return {
                "success": False,
                "message": "Training launcher not found. Please ensure src/infra/launch.py exists."
            }
        
        if not os.path.exists(dataset_path):
            return {
                "success": False,
                "message": f"Dataset path does not exist: {dataset_path}"
            }
        
        try:
            # This is a dry-run to test the command
            cmd = [
                sys.executable, "-m", "src.infra.launch",
                f"dataset.path={dataset_path}",
                "--help"  # Just show help to test if the module works
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": "Training launcher is accessible (dry run successful)",
                    "command": " ".join(cmd[:-1]),  # Remove --help
                    "dry_run": True
                }
            else:
                return {
                    "success": False,
                    "message": f"Training launcher failed: {result.stderr}",
                    "command": " ".join(cmd)
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error testing training launcher: {str(e)}"
            }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "status": self.training_status,
            "message": "Training status tracking",
            "logs_available": len(self.training_logs),
            "last_update": time.time()
        }

# Initialize manager
manager = SimpleTrainingManager()

def create_html_interface():
    """Create a simple HTML interface if Gradio is not available."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MonoX Training Interface</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .status {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
            .error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            .info {{ background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }}
            pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            button {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }}
            button:hover {{ background: #0056b3; }}
            .api-endpoint {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        </style>
        <script>
            async function testAPI() {{
                try {{
                    const response = await fetch('/system/info');
                    const data = await response.json();
                    document.getElementById('api-result').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    document.getElementById('api-status').className = 'status success';
                    document.getElementById('api-status').textContent = '✅ API is working! Response received.';
                }} catch (error) {{
                    document.getElementById('api-result').innerHTML = '<pre>Error: ' + error.message + '</pre>';
                    document.getElementById('api-status').className = 'status error';
                    document.getElementById('api-status').textContent = '❌ API Error: ' + error.message;
                }}
            }}
            
            async function getSystemInfo() {{
                const info = {json.dumps(manager.get_system_info(), indent=2)};
                document.getElementById('system-info').innerHTML = '<pre>' + JSON.stringify(info, null, 2) + '</pre>';
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🎨 MonoX Training Interface</h1>
            
            <div class="{'success' if HAS_GRADIO else 'warning'} status">
                {'✅ Gradio is available' if HAS_GRADIO else '⚠️ Gradio not available - using basic HTML interface'}
            </div>
            
            <div class="{'success' if HAS_TORCH else 'error'} status">
                {'✅ PyTorch is available' if HAS_TORCH else '❌ PyTorch not available - training will not work'}
            </div>
            
            <h2>System Information</h2>
            <button onclick="getSystemInfo()">Refresh System Info</button>
            <div id="system-info">
                <pre>{json.dumps(manager.get_system_info(), indent=2)}</pre>
            </div>
            
            <h2>API Test</h2>
            <p>Test if the JSON API is working correctly:</p>
            <button onclick="testAPI()">Test API</button>
            <div id="api-status" class="status info">Click "Test API" to check API functionality</div>
            <div id="api-result"></div>
            
            <h2>Available API Endpoints</h2>
            <div class="api-endpoint">
                <strong>GET /</strong> - This page
            </div>
            <div class="api-endpoint">
                <strong>GET /system/info</strong> - System information (JSON)
            </div>
            <div class="api-endpoint">
                <strong>GET /training/status</strong> - Training status (JSON)
            </div>
            <div class="api-endpoint">
                <strong>POST /training/start</strong> - Start training (JSON)
            </div>
            
            <h2>Setup Instructions</h2>
            <div class="info status">
                <strong>To fix the API issues:</strong><br>
                1. Ensure all dependencies are installed (see requirements.txt)<br>
                2. Make sure PyTorch and Gradio are available<br>
                3. Upload training data to the /dataset directory<br>
                4. Use the API endpoints above for JSON responses
            </div>
            
            <h2>Troubleshooting</h2>
            <div class="warning status">
                <strong>If you're getting HTML instead of JSON:</strong><br>
                • You're accessing the web interface URL instead of API endpoints<br>
                • Use <code>/system/info</code> instead of <code>/</code> for JSON<br>
                • Make sure to send proper HTTP headers for JSON requests
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def create_simple_server():
    """Create a simple HTTP server without external dependencies."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse
    
    class SimpleHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(create_html_interface().encode())
                
            elif self.path == '/system/info':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps(manager.get_system_info())
                self.wfile.write(response.encode())
                
            elif self.path == '/training/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps(manager.get_training_status())
                self.wfile.write(response.encode())
                
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not found"}).encode())
        
        def do_POST(self):
            if self.path == '/training/start':
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                try:
                    data = json.loads(post_data.decode()) if post_data else {}
                    result = manager.start_training(**data)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                    
                except Exception as e:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = {"success": False, "message": str(e)}
                    self.wfile.write(json.dumps(error_response).encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not found"}).encode())
        
        def log_message(self, format, *args):
            # Custom logging
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")
    
    return HTTPServer(('0.0.0.0', 7860), SimpleHandler)

def main():
    """Main function to start the appropriate server."""
    print("=" * 60)
    print("🎨 MonoX Training Interface Starting...")
    print("=" * 60)
    
    # Print system status
    info = manager.get_system_info()
    print(f"Python: {sys.version}")
    print(f"Gradio Available: {HAS_GRADIO}")
    print(f"PyTorch Available: {HAS_TORCH}")
    print(f"FastAPI Available: {HAS_FASTAPI}")
    
    if HAS_TORCH:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    
    print("=" * 60)
    
    # Start appropriate server
    if HAS_GRADIO:
        print("🚀 Starting Gradio interface...")
        # Import and run the full Gradio app
        try:
            from app import create_interface
            demo = create_interface()
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True
            )
        except Exception as e:
            print(f"❌ Gradio failed: {e}")
            print("🔄 Falling back to simple server...")
            server = create_simple_server()
            print("✅ Simple HTTP server running on http://0.0.0.0:7860")
            server.serve_forever()
    else:
        print("🔄 Gradio not available, starting simple HTTP server...")
        server = create_simple_server()
        print("✅ Simple HTTP server running on http://0.0.0.0:7860")
        print("📝 Access the web interface at: http://localhost:7860")
        print("🔗 API endpoints available at: /system/info, /training/status")
        server.serve_forever()

if __name__ == "__main__":
    main()