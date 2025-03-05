#!/usr/bin/env python3
# test_imports.py
import os
import importlib
import sys

def try_import(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

# Print the Python path
print("Python path:")
for p in sys.path:
    print(f" - {p}")

# Try importing various modules
print("\nTesting imports:")
try_import('permutation_weighting')
try_import('permutation_weighting.permutation_weighting')
try_import('permutation_weighting.permutation_weighting.models')
try_import('permutation_weighting.permutation_weighting.models.trainer_factory')
try_import('permutation_weighting.permutation_weighting.models.torch_trainer_factory')

# Print the package structure
print("\nPackage structure:")
for root, dirs, files in os.walk('permutation_weighting'):
    for file in files:
        if file.endswith('.py'):
            print(f" - {os.path.join(root, file)}")