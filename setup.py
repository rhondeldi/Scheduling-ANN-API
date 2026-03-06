#!/usr/bin/env python3
"""
Quick Start Script for ANN Model Setup
This script automates the initial setup process
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step_num, text):
    """Print formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 70)


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        return False


def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}/")


def main():
    """Main setup process"""
    
    print_header("ANN MODEL QUICK START SETUP")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python Version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    print_step(2, "Creating Project Directories")
    create_directories()
    
    # Step 3: Create virtual environment
    print_step(3, "Creating Virtual Environment")
    venv_created = run_command(
        "python -m venv venv",
        "Creating virtual environment 'venv'"
    )
    
    if not venv_created:
        print("Trying alternate command...")
        venv_created = run_command(
            "python3 -m venv venv",
            "Creating virtual environment with python3"
        )
    
    if not venv_created:
        print("✗ Failed to create virtual environment")
        sys.exit(1)
    
    # Determine activation command based on OS
    if sys.platform == "win32":
        activate_cmd = ".\\venv\\Scripts\\activate"
        pip_cmd = ".\\venv\\Scripts\\pip"
        python_cmd = ".\\venv\\Scripts\\python"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    print(f"\nTo activate virtual environment, run:")
    print(f"  {activate_cmd}")
    
    # Step 4: Install dependencies
    print_step(4, "Installing Python Dependencies")
    installed = run_command(
        f"{pip_cmd} install --upgrade pip",
        "Upgrading pip"
    )
    
    if installed:
        installed = run_command(
            f"{pip_cmd} install -r requirements.txt",
            "Installing requirements from requirements.txt"
        )
    
    if not installed:
        print("\n⚠️  You may need to install dependencies manually:")
        print(f"  {activate_cmd}")
        print(f"  pip install -r requirements.txt")
    
    # Step 5: Generate synthetic data
    print_step(5, "Generating Synthetic Training Data")
    data_generated = run_command(
        f"{python_cmd} data_collection.py",
        "Generating synthetic dataset"
    )
    
    # Step 6: Display model information
    print_step(6, "Displaying Model Architectures")
    run_command(
        f"{python_cmd} models.py",
        "Building and displaying model architectures"
    )
    
    # Final instructions
    print_header("SETUP COMPLETE!")
    
    print("Next Steps:")
    print("\n1. Activate virtual environment:")
    print(f"   {activate_cmd}")
    
    print("\n2. Train the fitness predictor model:")
    print("   python train_fitness_predictor.py")
    
    print("\n3. Start the API service:")
    print("   python api_service.py")
    
    print("\n4. Test the API:")
    print("   curl http://localhost:8000/health")
    
    print("\n5. Integrate with Go backend:")
    print("   - Copy go_integration_client.go to your Go project")
    print("   - Follow integration examples in the file")
    
    print("\nDocumentation:")
    print("   - Implementation Guide: ANN_IMPLEMENTATION_GUIDE.md")
    print("   - Quick Reference: README.md")
    print("   - API Docs (when running): http://localhost:8000/docs")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Setup failed with error: {e}")
        sys.exit(1)
