"""
WSGI entrypoint for Render deployment.
This file imports the Flask app from the correct location.
"""

import sys
import os

# Add insurance-chatbot to the Python path
sys.path.insert(0, os.path.abspath('insurance-chatbot'))

# Import the Flask app from the correct module
from src.api import app

# This is what Render will look for by default
if __name__ == "__main__":
    app.run()
