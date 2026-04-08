"""OpenEnv entry point — serves the FastAPI app."""

import uvicorn
from .server import app

__all__ = ["app", "main"]


def main():
    """Run the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
