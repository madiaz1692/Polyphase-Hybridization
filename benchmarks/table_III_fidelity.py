#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
from skimage import data
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
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

# --- Benchmark Dataset Generation ---
def get_test_dataset():
    """Load 3 natural images and 1 synthetic checkerboard at 512x512."""
    cam = data.camera() # 512x512
    
    ast = data.astronaut() # 512x512 RGB
    ast = cv2.cvtColor(ast, cv2.COLOR_RGB2GRAY)
    
    cof = data.coffee() # 400x600 RGB
    cof = cv2.cvtColor(cof, cv2.COLOR_RGB2GRAY)
    cof = cv2.resize(cof, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Natural Moon image
    mon = data.moon() # 512x512
            
    return {
        'Camera': cam,
        'Astronaut': ast,
        'Coffee': cof,
        'Moon': mon
    }


_DATASET = get_test_dataset()

def get_images(name):
    return {'native': _DATASET[name]}


# Standard fidelity protocol
RESOLUTIONS = [512, 1024, 2048]

def compute_metrics(orig, rt):
    h, w = orig.shape
    # Central 50% crop to avoid boundary artifacts
    edge_h, edge_w = h // 4, w // 4
    a = orig[edge_h:-edge_h, edge_w:-edge_w]
    b = rt[edge_h:-edge_h, edge_w:-edge_w]
    s = float(ssim(a, b, data_range=255))
    # Clip to avoid infinity if perfect
    diff = np.abs(a.astype(float) - b.astype(float))
    if np.max(diff) < 1e-6:
        p = 100.0 # Cap perfect PSNR at 100dB for averaging
    else:
        p = float(psnr(a, b, data_range=255))
    return s, p

def run_experiment():
    angles = np.arange(0, 46, 1) # 0 to 45 inclusive
    benchmark_images = ['Camera', 'Astronaut', 'Coffee', 'Moon']
    
    print("=" * 105)
    print(f"{' PH: TABLE III - HARMONIZED NATIVE FIDELITY ANALYSIS ':^105}")
    print(f"{' Comp. Angle Sweep (0-45°) | Dataset: {benchmark_images} (512x512) ':^105}")
    print("=" * 105)
    
    methods = [
        ('OpenCV Cubic', lambda img, ang: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2.0-0.5, img.shape[0]/2.0-0.5), ang, 1.0), (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)),
        ('PH-Fast (4x4)', lambda img, ang: ph.rotate(img, ang, mode='fast', beta=3, reshape=False, return_uint8=True)),
        ('SciPy Spline-3', lambda img, ang: scipy_rotate(img, ang, reshape=False, order=3, mode='reflect', prefilter=True)), 
        ('PH-HQ   (8x8)', lambda img, ang: ph.rotate(img, ang, mode='hq',   beta=5, reshape=False, return_uint8=True)),
        ('SciPy Spline-5', lambda img, ang: scipy_rotate(img, ang, reshape=False, order=5, mode='reflect', prefilter=True)),
        ('PH-SHQ  (12x12)', lambda img, ang: ph.rotate(img, ang, mode='shq',  beta=7, reshape=False, return_uint8=True)),
    ]

    print(f"{'Method':<18} | {'Camera':<9} | {'Astron.':<9} | {'Coffee':<9} | {'Checker.':<9} | {'GLOBAL PSNR':<12} | {'GLOBAL SSIM':<12}")
    print("-" * 115)

    results_for_paper = []

    for name, func in methods:
        all_metrics_p = []
        all_metrics_s = []
        
        sys.stdout.write(f"{name:<18} | ")
        sys.stdout.flush()
        
        for img_name in benchmark_images:
            img = get_images(img_name)['native']
            img_psnrs = []
            
            for ang in angles:
                out_rot = func(img, ang)
                rt = func(out_rot, -ang)
                s, p = compute_metrics(img, rt)
                img_psnrs.append(p)
                all_metrics_p.append(p)
                all_metrics_s.append(s)
            
            avg_img_p = np.mean(img_psnrs)
            sys.stdout.write(f"{avg_img_p:8.2f}  | ")
            sys.stdout.flush()
            
        grand_avg_p = np.mean(all_metrics_p)
        grand_avg_s = np.mean(all_metrics_s)
        print(f"{grand_avg_p:9.2f} dB | {grand_avg_s:10.4f}")
        results_for_paper.append((name, grand_avg_s, grand_avg_p))
    
    print("-" * 115)
    print("\n[SUMMARY FOR TABLE III]")
    for name, s, p in results_for_paper:
        print(f"{name:<18} & {s:.4f} & {p:.2f} \\\\")

if __name__ == "__main__":
    run_experiment()


