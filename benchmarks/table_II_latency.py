#!/usr/bin/env python3
"""
table_II_latency.py -- Computational Scalability: Latency (ms)
Ref: Polyphase Hybridization, Table II (Line 214)
Measures throughput across multiple resolutions.
"""

import numpy as np
import time
import os
import sys
import cv2
import platform
from scipy.ndimage import rotate as scipy_rotate
from skimage import data

# Ensure we can import libPH
try:
    import libPH.engine as ph
except ImportError:
    # Recursively find libPH
    curr = os.path.abspath(os.curdir)
    while curr != os.path.dirname(curr):
        if 'libPH' in os.listdir(curr):
            sys.path.append(curr)
            break
        curr = os.path.dirname(curr)
    import libPH.engine as ph

# --- Benchmark Protocol ---
# libPH/OpenCV: 100 reps + 5 preheats
# SciPy: 5 reps (due to high cost) + 1 preheat
RESOLUTIONS = [512, 1024, 2048]
NUM_PREHEAT = 5
NUM_RUN_PH_CV = 100
NUM_RUN_SCIPY = 5
TARGET_ANGLE = 22.5

def run_benchmark():
    print("=" * 80)
    print(f"{' PH: TABLE II - COMPUTATIONAL SCALABILITY ':^80}")
    print(f"{' Platform: ' + platform.processor() + ' (' + platform.machine() + ') ':^80}")
    print("=" * 80)
    print(f"{'Method':<30} | {'512^2 (ms)':<12} | {'1024^2 (ms)':<12} | {'2048^2 (ms)':<12}")
    print("-" * 80)
    
    methods = [
        ('OpenCV Bilinear', lambda img, ang: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2.0-0.5, img.shape[0]/2.0-0.5), ang, 1.0), (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR), NUM_RUN_PH_CV),
        ('OpenCV Cubic',    lambda img, ang: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2.0-0.5, img.shape[0]/2.0-0.5), ang, 1.0), (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC), NUM_RUN_PH_CV),
        ('SciPy Spline-3',  lambda img, ang: scipy_rotate(img, ang, reshape=False, order=3, mode='reflect', prefilter=True), NUM_RUN_SCIPY),
        ('SciPy Spline-5',  lambda img, ang: scipy_rotate(img, ang, reshape=False, order=5, mode='reflect', prefilter=True), NUM_RUN_SCIPY),
        ('PH-Fast (4x4)',   lambda img, ang: ph.rotate(img, ang, mode='fast', beta=3, return_uint8=True), NUM_RUN_PH_CV),
        ('PH-HQ (8x8)',     lambda img, ang: ph.rotate(img, ang, mode='hq',   beta=5, return_uint8=True), NUM_RUN_PH_CV),
        ('PH-SHQ (12x12)',  lambda img, ang: ph.rotate(img, ang, mode='shq',  beta=7, return_uint8=True), NUM_RUN_PH_CV),
    ]

    results = {m[0]: [] for m in methods}
    
    for res in RESOLUTIONS:
        # Prepare Image
        img_base = (data.camera())
        img = cv2.resize(img_base, (res, res))
        
        for name, func, num_run in methods:
            # 1. Preheat
            for _ in range(NUM_PREHEAT):
                _ = func(img, TARGET_ANGLE)
            
            # 2. Measurement
            t_start = time.perf_counter()
            for _ in range(num_run):
                _ = func(img, TARGET_ANGLE)
            t_end = time.perf_counter()
            
            avg_ms = (t_end - t_start) * 1000.0 / num_run
            results[name].append(avg_ms)

    # Print Final Table
    print(f"{'Method':<30} | {'512^2 (ms)':<12} | {'1024^2 (ms)':<12} | {'2048^2 (ms)':<12}")
    print("-" * 80)
    for name, _, _ in methods:
        ms_vals = results[name]
        print(f"{name:<30} | {ms_vals[0]:>11.2f} | {ms_vals[1]:>11.2f} | {ms_vals[2]:>11.2f}")

    print("-" * 80)
    # Calculate and show relative speedups/efficiency (using 2K resolution for peak saturation)
    print(f"{' libPH PERFORMANCE ANALYSIS (v. Baseline) ':^80}")
    print("-" * 80)
    
    # helper to get speedup
    def get_speedup(ph_name, base_name):
        base_ms = results[base_name][2] # 2K res
        ph_ms = results[ph_name][2]
        return base_ms / ph_ms

    speedup_fast = get_speedup('PH-Fast (4x4)', 'OpenCV Cubic')
    speedup_hq   = get_speedup('PH-HQ (8x8)',   'SciPy Spline-3')
    speedup_shq  = get_speedup('PH-SHQ (12x12)',  'SciPy Spline-5')

    print(f"PH-Fast (Bicubic) vs OpenCV Cubic    : {speedup_fast:>6.2f}x Faster")
    print(f"PH-HQ   (Quintic) vs SciPy Spline-3  : {speedup_hq:>6.2f}x Faster")
    print(f"PH-SHQ  (Septic)  vs SciPy Spline-5  : {speedup_shq:>6.2f}x Faster")

    print("-" * 80)
    print("Success: Performance Benchmarks generated with baseline parity analysis.")
    print("Note: Latency varies based on kernel support/memory fetch count.")
    print("Execution is memory-bandwidth bound; computational cost is transparent.")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
