#!/usr/bin/env python3
# install_dev.py

import os
import subprocess
import sys


def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    # Verify we're in the correct directory
    if not os.path.exists("setup.py"):
        print("Error: This script must be run from the project root directory")
        return False

    # Check for virtual environment
    in_venv = False
    if "VIRTUAL_ENV" in os.environ:
        in_venv = True
    elif "CONDA_PREFIX" in os.environ:
        in_venv = True
    elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        in_venv = True

    if not in_venv:
        print("Warning: It seems you're not in a virtual environment.")
        if input("Continue anyway? [y/n]: ").lower() != 'y':
            return False

    # Install PyTorch
    print("Installing PyTorch...")
    if not run_command("pip install torch"):
        return False

    # Uninstall any existing version
    print("Removing any existing package installation...")
    run_command("pip uninstall -y permutation_weighting")

    # Install in development mode
    print("Installing package in development mode...")
    if not run_command("pip install -e ."):
        return False

    # Install test dependencies
    print("Installing test dependencies...")
    if not run_command("pip install pytest"):
        return False

    print("\nSetup complete!")
    print("Try running tests with: pytest -xvs permutation_weighting/tests/")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)