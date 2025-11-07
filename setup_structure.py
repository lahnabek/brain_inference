"""
setup_structure.py

This script checks for the presence of required directories and files 
in the project root and creates any missing directories automatically.

Usage:
    Run this script from anywhere, it will locate the project root 
    (two levels up from this script's location) and ensure the project 
    folder structure and key files are in place.

This is useful for initializing or verifying the project environment 
before running other code.

PROJECT_DIRS lists the essential directories to exist.
REQUIRED_FILES lists essential files that should be present.
"""

from pathlib import Path

PROJECT_DIRS = [
    "data_tvb/connectome_TVB",
    "data_tvb/real_connectome_for_tvb",
    "data_tvb/kernel",
    "generated_data",
    "inference",
    "notebooks",
    "original_dataset",
    "results",
    "results/synch",
    "save_file",
    "simulation_file",
    "simulation_file/parameter",
    "test",
]

REQUIRED_FILES = [
    "requirements.txt",
    "setup.py"
]

def create_structure():
    # Define root as two levels above this script file location
    root = Path(__file__).resolve().parent.parent
    print(f"🔍 [ROOT DIR] {root}")

    for rel_dir in PROJECT_DIRS:
        path = root / rel_dir
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f" Created directory: {path}")
        else:
            print(f" Exists: {path}")

    for rel_file in REQUIRED_FILES:
        path = root / rel_file
        if not path.exists():
            print(f" MISSING FILE: {path}")
        else:
            print(f" File OK: {path}")

if __name__ == "__main__":
    create_structure()
