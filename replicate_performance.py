"""
Polyphase Hybridization (PH) - Master Replication Script
Polyphase Hybridization Validation Suite
Authors: Maria S. Diaz-Gonzalez, Jose L. Gonzalez-Mora
"""

import os
import sys
import subprocess
import time

def run_script(path):
    print(f"\n>>> Executing: {os.path.basename(path)}...")
    print("-" * 40)
    try:
        # Run with current python interpreter and include current dir in path
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        
        result = subprocess.run([sys.executable, path], env=env)
        if result.returncode != 0:
            print(f"[ERROR] Script {path} failed with return code {result.returncode}")
    except Exception as e:
        print(f"[ERROR] Failed to run {path}: {e}")
    print("-" * 40)

def main():
    print("=" * 80)
    print(f"{' PH: MASTER REPLICATION SUITE ':^80}")
    print(f"{' Polyphase Hybridization Framework ':^80}")
    print("=" * 80)
    
    # Requirement Check
    missing = []
    try: import numpy; print("[OK] NumPy found.")
    except ImportError: missing.append("numpy")
    try: import cv2; print("[OK] OpenCV found.")
    except ImportError: missing.append("opencv-python")
    try: import scipy; print("[OK] SciPy found.")
    except ImportError: missing.append("scipy")
    try: import skimage; print("[OK] Scikit-Image found.")
    except ImportError: missing.append("scikit-image")
    
    if missing:
        print("\n[CRITICAL] Missing dependencies:", ", ".join(missing))
        print("Please run: pip install numpy opencv-python scipy scikit-image")
        return

    # Table Execution Order
    tables = [
        "benchmarks/table_I_anchors.py",
        "benchmarks/table_II_latency.py",
        "benchmarks/table_III_fidelity.py",
        "benchmarks/table_IV_stability.py",
        "benchmarks/table_V_iterative.py"
    ]
    
    start_time = time.time()
    for table in tables:
        if os.path.exists(table):
            run_script(table)
        else:
            print(f"[SKIP] {table} not found.")
            
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 80)
    print(f"{' REPLICATION COMPLETE ':^80}")
    print(f"Total processing time: {total_time:.1f} minutes")
    print("=" * 80)

if __name__ == "__main__":
    main()
