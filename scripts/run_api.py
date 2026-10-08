"""Start the API on your own computer. Stop it with the red square in PyCharm (or Ctrl+C).

Then open http://127.0.0.1:8000/docs in your browser to try every endpoint.
"""
import uvicorn

from src.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)