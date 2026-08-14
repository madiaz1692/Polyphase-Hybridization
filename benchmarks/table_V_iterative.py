#!/usr/bin/env python3
"""
table_V_iterative.py -- Iterative Robustness Suite across Dual Angular Regimes
Ref: Polyphase Hybridization (Array Journal), Table 5
Benchmarks two complementary regimes:
  1. Fine Micro-Step Regime: 360 consecutive 1.0° steps (evaluating Step 90 and Step 360).
  2. Anchor-Aligned Regime: 72 consecutive 5.0° steps (evaluating Step 18 [90°] and Step 72 [360°]).
"""

import cv2
import numpy as np
import os
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import data
from scipy.ndimage import rotate as scipy_rotate

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
SAVE_DEBUG_IMGS = False
DEBUG_DIR = "debug_iterative"

def add_visual_flag(img, text):
    """Draws a label box in the corner of the image."""
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
    # Central 512x512 ROI evaluation on 800x800 canvas
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

def run_regime(img_800, step_deg, total_steps, milestone_steps):
    methods = [
        ('OpenCV Cubic', lambda i, a: cv2.warpAffine(i, cv2.getRotationMatrix2D((i.shape[1]/2.0-0.5, i.shape[0]/2.0-0.5), a, 1.0), (i.shape[1], i.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)),
        ('PH-Fast (4x4)', lambda i, a: ph.rotate(i, a, mode='fast', beta=3, reshape=False, return_uint8=True)),
        ('SciPy Spline-3', lambda i, a: scipy_rotate(i, a, reshape=False, order=3, mode='reflect', prefilter=True)),
        ('PH-HQ (8x8)', lambda i, a: ph.rotate(i, a, mode='hq', beta=5, reshape=False, return_uint8=True)),
        ('SciPy Spline-5', lambda i, a: scipy_rotate(i, a, reshape=False, order=5, mode='reflect', prefilter=True)),
        ('PH-SHQ (12x12)', lambda i, a: ph.rotate(i, a, mode='shq', beta=7, reshape=False, return_uint8=True)),
    ]

    states = []
    for m_name, func in methods:
        init_img = img_800.copy().astype(np.uint8)
        states.append({'name': m_name, 'func': func, 'curr': init_img, 'results': {}})

    for step in range(1, total_steps + 1):
        cum_angle = step * step_deg
        for s in states:
            s['curr'] = s['func'](s['curr'], step_deg)
            
            if step in milestone_steps:
                # Reference: exact np.rot90 for 90/180/270/360 or single-pass ground truth
                if int(round(cum_angle)) % 360 == 0:
                    ref = img_800
                elif int(round(cum_angle)) == 90:
                    ref = np.rot90(img_800, k=1)
                else:
                    ref = scipy_rotate(img_800, cum_angle, reshape=False, order=5, mode='reflect', prefilter=True)
                
                p, ss = compute_metrics(ref, s['curr'])
                s['results'][step] = (p, ss)

    return {s['name']: s['results'] for s in states}

def run_iterative_test():
    img = data.camera()
    pad = 144
    img_800 = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    print("=" * 125)
    print(f"{' PH: TABLE 5 - ITERATIVE ROBUSTNESS SUITE (DUAL-REGIME EVALUATION) ':^125}")
    print(f"{' (1.0 deg Micro-Steps vs 5.0 deg Anchor-Aligned Orbit | Canvas: 800x800 | ROI: 512x512) ':^125}")
    print("=" * 125)

    print("Running Regime 1: Fine Micro-Steps (DeltaTheta = 1.0 deg, 360 steps)...")
    res_1deg = run_regime(img_800, step_deg=1.0, total_steps=360, milestone_steps=[90, 360])

    print("Running Regime 2: Anchor-Aligned Steps (DeltaTheta = 5.0 deg, 72 steps)...")
    res_5deg = run_regime(img_800, step_deg=5.0, total_steps=72, milestone_steps=[18, 72])

    print("\n" + "=" * 125)
    print(f"{'':<18} | {'Fine Micro-Step Regime (DeltaTheta = 1.0 deg)':^48} | {'Anchor-Aligned Regime (DeltaTheta = 5.0 deg)':^48}")
    print(f"{'Method':<18} | {'Step 90 (90deg)':^22} | {'Step 360 (360deg)':^23} | {'Step 18 (90deg)':^22} | {'Step 72 (360deg)':^23}")
    print(f"{'':<18} | {'PSNR (dB) / SSIM':^22} | {'PSNR (dB) / SSIM':^23} | {'PSNR (dB) / SSIM':^22} | {'PSNR (dB) / SSIM':^23}")
    print("-" * 125)

    method_names = ['OpenCV Cubic', 'PH-Fast (4x4)', 'SciPy Spline-3', 'PH-HQ (8x8)', 'SciPy Spline-5', 'PH-SHQ (12x12)']

    for name in method_names:
        r1_90 = res_1deg[name].get(90, (0.0, 0.0))
        r1_360 = res_1deg[name].get(360, (0.0, 0.0))
        r5_18 = res_5deg[name].get(18, (0.0, 0.0))
        r5_72 = res_5deg[name].get(72, (0.0, 0.0))

        str_r1_90 = f"{r1_90[0]:>5.2f} / {r1_90[1]:.3f}"
        str_r1_360 = f"{r1_360[0]:>5.2f} / {r1_360[1]:.3f}"
        str_r5_18 = f"{r5_18[0]:>5.2f} / {r5_18[1]:.3f}"
        str_r5_72 = f"{r5_72[0]:>5.2f} / {r5_72[1]:.3f}"

        print(f"{name:<18} | {str_r1_90:^22} | {str_r1_360:^23} | {str_r5_18:^22} | {str_r5_72:^23}")

    print("=" * 125)

if __name__ == "__main__":
    run_iterative_test()
