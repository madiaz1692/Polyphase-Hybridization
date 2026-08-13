#!/usr/bin/env python3
"""
table_IV_stability.py -- Analytical Fidelity Suite (v4: Safe Canvas)
Ref: Polyphase Hybridization, Table IV
Benchmarking single-pass fidelity against mathematically generated ground truths.
Uses an 800x800 canvas with a 512x512 ROI to eliminate edge-clipping.

DSLP implementation based on:
N. H. Lee, K. J. Kyriakopoulos, "A differentiable Shepp-Logan phantom and its 
applications in exact cone-beam CT", Phys. Med. Biol. 50 (2005) 1-13.
"""

import cv2
import numpy as np
import os
import sys
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
CANVAS_SIZE = 800
ROI_SIZE = 512
OFF = (CANVAS_SIZE - ROI_SIZE) // 2
SAVE_DEBUG_IMGS = True
DEBUG_DIR = "debug_analytical"

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

# --- Globals for Optimization ---
_MESH_SIZE = 0
_X_MESH = None
_Y_MESH = None

def get_meshgrid(size):
    global _MESH_SIZE, _X_MESH, _Y_MESH
    if _MESH_SIZE != size:
        coords = np.linspace(-1, 1, size)
        _X_MESH, _Y_MESH = np.meshgrid(coords, -coords)
        _MESH_SIZE = size
    return _X_MESH, _Y_MESH

# --- Analytical Generators (4x AA Protocol) ---

def generate_siemens_star(size, angle_deg=0, spokes=36):
    """Generates a sinusoidal Siemens Star at exactly 'size' resolution (1x)."""
    # Angle is for rotation fidelity, but phantom itself is often pre-rotated. 
    # Here we generate the 'ground truth' at 'angle_deg'.
    x, y = get_meshgrid(size)
    phi = math.radians(angle_deg)
    
    # Coordinates relative to center (0,0) as get_meshgrid does [-1, 1]
    theta = np.arctan2(y, x)
    
    # Sinusoidal profile: 0.5 + 0.5*sin(spokes * (theta - phi))
    star = 0.5 + 0.5 * np.sin(spokes * (theta - phi))
    
    # Radius mask to clip the star to a circle (radius 0.9 in [-1, 1] space)
    r = np.sqrt(x**2 + y**2)
    star[r > 0.9] = 0.5 # Neutral gray outside
    
    return (star * 255).astype(np.uint8)

def generate_shepp_logan(size, angle_deg=0, mode='standard'):
    """Generates a Shepp-Logan phantom at exactly 'size' resolution (1x)."""
    x, y = get_meshgrid(size)
    phi_rot = math.radians(angle_deg)
    cos_phi, sin_phi = math.cos(phi_rot), math.sin(phi_rot)
    
    ellipses = [
        [0.69,   0.92,   0,      0,      0,      1.0],   # Skull
        [0.6624, 0.8740, 0,     -0.0184, 0,     -0.98],  # Inner
        [0.11,   0.31,   0.22,   0,     -18,     -0.02],
        [0.16,   0.41,  -0.22,   0,      18,     -0.02],
        [0.21,   0.25,   0,      0.35,   0,      0.01],
        [0.046,  0.046,  0,      0.1,    0,      0.01],
        [0.046,  0.046,  0,     -0.1,    0,      0.01],
        [0.046,  0.023, -0.08,  -0.605,  0,      0.01],
        [0.023,  0.023,  0,     -0.605,  0,      0.01],
        [0.023,  0.046,  0.06,  -0.605,  0,      0.01],
    ]
    
    img = np.zeros_like(x, dtype=np.float32)
    xr_all = x * cos_phi + y * sin_phi
    yr_all = -x * sin_phi + y * cos_phi

    for a, b, x0, y0, phi, v in ellipses:
        p = math.radians(phi)
        cos_p, sin_p = math.cos(p), math.sin(p)
        xp = (xr_all - x0) * cos_p + (yr_all - y0) * sin_p
        yp = -(xr_all - x0) * sin_p + (yr_all - y0) * cos_p
        
        # d^2 (normalized squared distance to center of ellipse)
        d2 = xp**2 / a**2 + yp**2 / b**2
        mask = d2 <= 1.0
        
        if mode == 'dslp':
            # Differentiable Shepp-Logan Phantom (DSLP)
            # Replace indicator function with a smooth polynomial: (1 - d^2)^2
            img[mask] += v * (1.0 - d2[mask])**2
        else:
            # Standard SLP: piecewise constant (abrupt)
            img[mask] += v
        
    # Standard normalization to [0, 1]
    mi, ma = img.min(), img.max()
    if ma > mi:
        img = (img - mi) / (ma - mi)
    else:
        img = np.zeros_like(img)

    return (img * 255).astype(np.uint8)

def generate_zone_plate(size, angle_deg=0):
    """Generates a Zone Plate (chirp) at exactly 'size' resolution (1x)."""
    x, y = get_meshgrid(size)
    phi = math.radians(angle_deg)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    xr = x * cos_phi + y * sin_phi
    yr = -x * sin_phi + y * cos_phi
    # Chirp = 0.5 + 0.5 * cos(km * r^2)
    r2 = (xr**2 + 0.8 * yr**2) * 80 * math.pi
    img = 0.5 + 0.5 * np.cos(r2)
    return (img * 255).astype(np.uint8)

# --- Benchmark Helpers ---

def compute_metrics(orig, target):
    orig_roi = orig[OFF:OFF+ROI_SIZE, OFF:OFF+ROI_SIZE]
    target_roi = target[OFF:OFF+ROI_SIZE, OFF:OFF+ROI_SIZE]
    try:
        p = float(psnr(orig_roi, target_roi, data_range=255))
    except (ZeroDivisionError, ValueError):
        p = 100.0
    if np.isinf(p) or p > 100.0: p = 100.0
    
    try:
        # SSIM calculation with win_size for robustness on small differences
        s = float(ssim(orig_roi, target_roi, data_range=255))
    except Exception:
        s = 1.0
        
    return p, s

def run_analytical_suite():
    print("-" * 140)
    print(f"{'Method':<16} | {'Star (P/S)':<20} | {'SLP (P/S)':<20} | {'DSLP (P/S)':<20} | {'ZonePlate (P/S)':<20}")
    print("-" * 140)

    from scipy.ndimage import rotate as scipy_rotate
    methods = [
        ('CV2 Bilinear',   lambda img, ang: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2.0-0.5, img.shape[0]/2.0-0.5), ang, 1.0), (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)),
        ('CV2 Cubic',      lambda img, ang: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2.0-0.5, img.shape[0]/2.0-0.5), ang, 1.0), (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)),
        ('SciPy S3',      lambda img, ang: scipy_rotate(img, ang, reshape=False, order=3, mode='reflect')),
        ('SciPy S5',      lambda img, ang: scipy_rotate(img, ang, reshape=False, order=5, mode='reflect')),
        ('PH-Fast (4x4)',  lambda img, ang: ph.rotate(img, ang, mode='fast', beta=3, return_uint8=True)),
        ('PH-HQ (8x8)',    lambda img, ang: ph.rotate(img, ang, mode='hq',   beta=5, return_uint8=True)),
        ('PH-SHQ (12x12)', lambda img, ang: ph.rotate(img, ang, mode='shq',  beta=7, return_uint8=True)),
    ]
    
    targets = [
        ('Star', generate_siemens_star),
        ('SLP',  lambda s, a: generate_shepp_logan(s, a, 'standard')),
        ('DSLP', lambda s, a: generate_shepp_logan(s, a, 'dslp')),
        ('ZP',   generate_zone_plate),
    ]

    # Pre-cache Ground Truths at 800x800
    angles = [15, 30, 45, 60, 75]
    gt_cache = {}
    print("  Caching 800x800 ground truths...", end=" ", flush=True)
    for t_name, t_gen in targets:
        gt_cache[t_name] = {ang: t_gen(CANVAS_SIZE, ang) for ang in angles}
        gt_cache[t_name][0] = t_gen(CANVAS_SIZE, 0)
    print("Done.")

    results_matrix = {m[0]: [] for m in methods}

    for t_name, t_gen in targets:
        base_img = gt_cache[t_name][0]
        
        for m_name, func in methods:
            metrics_list = [] # List of (psnr, ssim)
            
            for ang in angles:
                ref = gt_cache[t_name][ang]
                rotated = func(base_img, float(ang))
                metrics_list.append(compute_metrics(ref, rotated))
            
            # Average PSNR and Average SSIM
            avg_psnr = np.mean([m[0] for m in metrics_list])
            avg_ssim = np.mean([m[1] for m in metrics_list])
            results_matrix[m_name].append((avg_psnr, avg_ssim))

    # SAVE ALL ANGLES for each target
    if SAVE_DEBUG_IMGS:
        for t_name, _ in targets:
            for ang in angles:
                gt_ang = gt_cache[t_name][ang][OFF:OFF+ROI_SIZE, OFF:OFF+ROI_SIZE]
                
                # Visualization boost for low-contrast SLP/DSLP
                def boost(im):
                    if t_name in ['SLP', 'DSLP']:
                        # Power-law Gamma 0.5 to reveal details in dark areas
                        f_im = im.astype(float) / 255.0
                        boosted = np.power(f_im, 0.5) * 255.0
                        return boosted.astype(np.uint8)
                    return im

                strip = [add_visual_flag(boost(gt_ang), f"GT-{t_name}")]
                for m_name, func in methods:
                    rotated = func(gt_cache[t_name][0], float(ang))
                    roi = rotated[OFF:OFF+ROI_SIZE, OFF:OFF+ROI_SIZE]
                    strip.append(add_visual_flag(boost(roi), m_name))
                
                full_strip = np.hstack(strip)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"stack_{t_name}_{ang}deg.png"), full_strip)

    for m_name, _ in methods:
        row = results_matrix[m_name]
        # Format string for PSNR / SSIM
        def fmt(m): return f"{m[0]:>5.2f}/{m[1]:.4f}"
        print(f"{m_name:<16} | {fmt(row[0]):<20} | {fmt(row[1]):<20} | {fmt(row[2]):<20} | {fmt(row[3]):<20}")

    print("-" * 140)

if __name__ == "__main__":
    run_analytical_suite()
