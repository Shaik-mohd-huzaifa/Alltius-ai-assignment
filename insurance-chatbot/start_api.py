"""
Quick start script for the Insurance Chatbot API server.
This script simplifies launching the API server with optional configuration.
"""

import os
import argparse
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if the OpenAI API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("\nWARNING: OpenAI API key not set or using default value.")
        print("    Update the .env file with your actual API key.")
        return False
    return True

def main():
    """Parse command line arguments and start the API server."""
    parser = argparse.ArgumentParser(description="Start the Insurance Chatbot API server")
    
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host to run the API server on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, 
                        help="Port to run the API server on (default: 5000)")
    parser.add_argument("--debug", action="store_true", 
                        help="Run the API server in debug mode")
    
    args = parser.parse_args()
    
    # Check for API key
    check_api_key()
    
    # Print startup message
    print("\n" + "=" * 70)
    print(" Insurance Chatbot API Server ".center(70, "="))
    print("=" * 70)
    print(f"Starting API server on http://{args.host}:{args.port}")
    
    if args.host == "0.0.0.0":
        print(f"Access locally at: http://localhost:{args.port}")
    
    print("\nAPI Endpoints:")
    print(f"  - Chat:   http://localhost:{args.port}/api/chat")
    print(f"  - Ingest: http://localhost:{args.port}/api/ingest")
    print(f"  - Health: http://localhost:{args.port}/api/health")
    
    print("\nConnect your React frontend to these endpoints")
    print("=" * 70 + "\n")
    
    # Start the API server
    api_path = os.path.join(os.path.dirname(__file__), "src", "api.py")
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["FLASK_APP"] = api_path
    env["FLASK_ENV"] = "development" if args.debug else "production"
    
    cmd = [
        "python", api_path,
        "--host", args.host,
        "--port", str(args.port),
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nShutting down API server...")

if __name__ == "__main__":
    main()
