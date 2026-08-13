#!/usr/bin/env python3
"""
table_V_iterative.py -- Iterative Robustness Suite with Visual Stacking)
Ref: Polyphase Hybridization, Table V
Benchmarks 90 and 360 1-degree-step cumulative rotation.
Isolates quantization error using float32/uint8 comparison.
"""

import cv2
import numpy as np
import os
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import data

# Ensure we can import libPH
try:
    import libPH.engine as ph
except ImportError:
    curr = os.path.abspath(os.curdir)
    while curr != os.path.dirname(curr):
        if 'libPH' in os.listdir(curr):
            sys.path.append(curr)
            break
        curr = os.path.dirname(curr)
    import libPH.engine as ph

# --- Configuration ---
SAVE_DEBUG_IMGS = False # Set to True to enable per-step/per-angle image exports
DEBUG_DIR = "debug_iterative"

def add_visual_flag(img, text):
    """Draws a professional label box in the corner of the image."""
    res = img.copy()
    if res.ndim == 2:
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(res, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(res, text, (5, th + 5), font, font_scale, (255, 255, 255), thickness)
    return res

if SAVE_DEBUG_IMGS:
    os.makedirs(DEBUG_DIR, exist_ok=True)

def compute_metrics(orig, target):
    # Same ROI protocol as engine regression
    pad = (orig.shape[0] - 512) // 2
    o = orig[pad:-pad, pad:-pad]
    t = target[pad:-pad, pad:-pad]
    
    if t.dtype != np.uint8:
        t = np.clip(t, 0, 255).astype(np.uint8)
    if o.dtype != np.uint8:
        o = np.clip(o, 0, 255).astype(np.uint8)

    p_val = float(psnr(o, t, data_range=255))
    s_val = float(ssim(o, t, data_range=255))
    return p_val, s_val

def run_iterative_test():
    img = data.camera()
    # 800x800 Safe Canvas via Reflection
    pad = 144
    img_800 = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    print("=" * 115)
    print(f"{' PH: TABLE V - ITERATIVE ROBUSTNESS SUITE ':^115}")
    print(f"{' (90 1.0 deg steps | Canvas: 800x800 | ROI: 512x512) ':^115}")
    print("=" * 115)

    from scipy.ndimage import rotate as scipy_rotate
    
    # Rotator registry (Method Name, Dtype, Func)
    methods = [
        ('CV2 Bilinear', lambda i, a: cv2.warpAffine(i, cv2.getRotationMatrix2D((i.shape[1]/2.0-0.5, i.shape[0]/2.0-0.5), a, 1.0), (i.shape[1], i.shape[0]), flags=cv2.INTER_LINEAR)),
        ('CV2 Cubic', lambda i, a: cv2.warpAffine(i, cv2.getRotationMatrix2D((i.shape[1]/2.0-0.5, i.shape[0]/2.0-0.5), a, 1.0), (i.shape[1], i.shape[0]), flags=cv2.INTER_CUBIC)),
        ('SciPy Spline-3', lambda i, a: scipy_rotate(i, a, reshape=False, order=3, mode='reflect', prefilter=True)),
        ('SciPy Spline-5', lambda i, a: scipy_rotate(i, a, reshape=False, order=5, mode='reflect', prefilter=True)),
        ('PH-Fast (4x4)', lambda i, a: ph.rotate(i, a, mode='fast', beta=3, return_uint8=True)),
        ('PH-HQ (8x8)', lambda i, a: ph.rotate(i, a, mode='hq',   beta=5, return_uint8=True)),
        ('PH-SHQ (12x12)', lambda i, a: ph.rotate(i, a, mode='shq',  beta=7, return_uint8=True)),
    ]

    # Persistent states for iterative simulation
    states = []
    for m_name, func in methods:
        init_img = img_800.copy().astype(np.uint8)
        states.append({'name': m_name, 'func': func, 'curr': init_img, 'results': {}})

    milestones = [90, 360]
    debug_milestones = list(range(15, 361, 15))

    for step in range(1, 361):
        for s in states:
            s['curr'] = s['func'](s['curr'], 1.0) # Accumulate 1 degree
            
            if step in milestones:
                # Select correct reference
                if step == 90:
                    ref = np.rot90(img_800, k=1)
                else: # 360
                    ref = img_800
                p, ss = compute_metrics(ref, s['curr'])
                s['results'][step] = (p, ss)
        
        # Periodic visual stacking for visualization
        if SAVE_DEBUG_IMGS and step in debug_milestones:
            strips = []
            for s in states:
                # Convert to u8 for saving
                img_to_save = s['curr']
                if img_to_save.dtype != np.uint8:
                    img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
                # Crop ROI 512
                roi = img_to_save[pad:-pad, pad:-pad]
                # Label
                label = f"{s['name']}"
                strips.append(add_visual_flag(roi, label))
            
            full_strip = np.hstack(strips)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"stack_all_step_{step}.png"), full_strip)

    print(f"{'Method':<20} | {'90deg (PSNR/SSIM)':^18} | {'360deg (PSNR/SSIM)':^18}")
    print("-" * 70)
    for s in states:
        m90 = s['results'].get(90, (0, 0))
        m360 = s['results'].get(360, (0, 0))
        print(f"{s['name']:<20} | {m90[0]:>6.2f} / {m90[1]:>5.3f} | {m360[0]:>6.2f} / {m360[1]:>5.3f}")

    print("-" * 70)

if __name__ == "__main__":
    run_iterative_test()
