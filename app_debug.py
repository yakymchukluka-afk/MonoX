#!/usr/bin/env python3
"""
MonoX Debug Version - Test API Routes
"""

import os
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="MonoX Debug", description="Debug version to test API routes")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MonoX Debug</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; margin: 10px; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>🎨 MonoX Debug</h1>
        <p>Testing API route registration</p>
        <button onclick="testAPI()">Test API</button>
        <button onclick="testHealth()">Test Health</button>
        
        <script>
            async function testAPI() {
                try {
                    const response = await fetch('/api/test');
                    const result = await response.json();
                    alert('API Test Success: ' + result.message);
                } catch (e) {
                    alert('API Test Error: ' + e.message);
                }
            }
            
            async function testHealth() {
                try {
                    const response = await fetch('/health');
                    const result = await response.json();
                    alert('Health Success: ' + result.message);
                } catch (e) {
                    alert('Health Error: ' + e.message);
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Debug version is working",
        "timestamp": time.time()
    }

@app.get("/api/test")
async def test_api():
    """Test API endpoint."""
    return {
        "message": "API routes are working correctly",
        "status": "success",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)