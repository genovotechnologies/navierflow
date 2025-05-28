import os
import shutil
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def init_project():
    """Initialize project structure"""
    # Create main directories
    directories = [
        "navierflow/core/eulerian",
        "navierflow/core/lbm",
        "navierflow/gui",
        "navierflow/utils",
        "configs",
        "data/raw",
        "data/processed",
        "outputs",
        "tests"
    ]
    
    for directory in directories:
        create_directory(directory)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_locations = [
        "navierflow",
        "navierflow/core",
        "navierflow/core/eulerian",
        "navierflow/core/lbm",
        "navierflow/gui",
        "navierflow/utils"
    ]
    
    for location in init_locations:
        init_file = Path(location) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"Created: {init_file}")

    print("\nProject structure initialized successfully!")

if __name__ == "__main__":
    init_project() 