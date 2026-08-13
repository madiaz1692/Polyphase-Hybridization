"""
Polyphase Hybridization (PH) Architecture
Standardized Python Engine (V3-Core)
Authors: Maria S. Diaz-Gonzalez, Jose L. Gonzalez-Mora
"""
import numpy as np
import ctypes
import os
import math
import platform

# -- Library Discovery --
current_dir = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(current_dir, "core")
WEIGHTS_DIR = os.path.join(current_dir, "weights")

def get_dll_path():
    sys_name = platform.system().lower()
    machine = platform.machine().lower()

    if sys_name == "windows":
        return os.path.join(CORE_DIR, "libph_x86.dll")
    
    # Manual override for testing
    if os.environ.get("PH_FORCE_AVX2") == "1":
        return os.path.join(CORE_DIR, "libph_x86.so")
    
    # Linux / ARM support
    if "aarch64" in machine or "arm64" in machine:
        return os.path.join(CORE_DIR, "libph_sve.so")
    else:
        # Check for AVX-512 specialized binary
        avx512_path = os.path.join(CORE_DIR, "libph_avx512.so")
        if os.path.exists(avx512_path):
            return avx512_path
        # Default to x86_64 AVX2 on Linux
        return os.path.join(CORE_DIR, "libph_x86.so")

DLL_PATH = get_dll_path()

# -- Anchor Selection Logic --
def find_anchor(target_angle_deg):
    """Finds the best Farey anchor for the target angle (25 <= A <= 50)."""
    best_anchor = (5, 0)
    min_diff = float('inf')
    target_rad = abs(math.radians(target_angle_deg))
    for x in range(1, 13):
        for y in range(0, 13):
            if math.gcd(x, y) == 1:
                A = x*x + y*y
                if A < 25 or A > 50: continue
                angle = math.atan2(y, x)
                diff = abs(angle - target_rad)
                if diff < min_diff:
                    min_diff = diff
                    best_anchor = (x, y)
    res_xr, res_yr = best_anchor
    if target_angle_deg < 0: res_yr = -res_yr
    return (res_xr, res_yr), math.degrees(math.atan2(res_yr, res_xr))

def get_rotated_size(w, h, angle_deg):
    rad = math.radians(abs(angle_deg))
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    new_w = int(w * cos_t + h * sin_t + 1.0)
    new_h = int(w * sin_t + h * cos_t + 1.0)
    if new_w % 2 != (w % 2): new_w += 1
    if new_h % 2 != (h % 2): new_h += 1
    return new_w, new_h

# -- Weight Loading/Synthesis Logic --
_lut_cache = {}
def get_lut(mode, beta, anchor_A):
    key = (mode, beta, anchor_A)
    if key in _lut_cache: return _lut_cache[key]
    
    fname = f"{mode}_beta{beta}_A{anchor_A}.npy"
    path = os.path.join(WEIGHTS_DIR, fname)
    if os.path.exists(path):
        lut = np.load(path)
    else:
        # Fallback to dynamic synthesis if available
        try:
            from .synthesis import synthesize_lut
            lut = synthesize_lut(mode=mode, beta=beta, num_phases=anchor_A)
        except ImportError:
            raise RuntimeError(f"Weights {fname} not found and synthesis.py missing.")
    
    flat = lut.astype(np.float32).flatten()
    _lut_cache[key] = flat
    return flat

# -- Optimized Backend --
class PHProcessor:
    def __init__(self):
        if not os.path.exists(DLL_PATH):
            # If precompiled binary is missing, we might be only in scalar mode if implemented
            pass 
        
        self.lib = None
        try:
            self.lib = ctypes.CDLL(DLL_PATH)
        except Exception:
            # print("PH: Warning - Optimized binary not found or failed to load. Using scalar fallback.")
            pass

        self.common_argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]

    def select_kernel(self, mode):
        if mode == 'shq': base = "ph_rotate_shq_12"
        elif mode == 'hq': base = "ph_rotate_hq"
        else: base = "ph_rotate_fast"
        
        if self.lib:
            # Try base name, then platform-specific suffixes
            for name in [base, f"{base}_avx512", f"{base}_avx2", f"{base}_sve"]:
                func = getattr(self.lib, name, None)
                if func:
                    func.argtypes = self.common_argtypes
                    func.restype = None
                    return func
        
        raise AttributeError(f"Kernel {base} not found. Ensure the optimized binary is in 'core/'.")

    def rotate(self, image, angle_deg, mode='hq', beta=None, reshape=False, preheated_lut=None, return_uint8=False):
        norm_angle = angle_deg % 360
        if norm_angle == 0: return image.copy()
        if norm_angle in [90, 180, 270]: return np.rot90(image, k=norm_angle//90).copy()

        if beta is None:
            beta = 7 if mode == 'shq' else (5 if mode == 'hq' else 3)

        src_h, src_w = image.shape
        dst_w, dst_h = get_rotated_size(src_w, src_h, angle_deg) if reshape else (src_w, src_h)
        sx, sy = (src_w - 1) * 0.5, (src_h - 1) * 0.5
        dx, dy = (dst_w - 1) * 0.5, (dst_h - 1) * 0.5
        
        ph_angle = -angle_deg
        (xr, yr), anchor_angle = find_anchor(ph_angle)
        delta_theta = (ph_angle - anchor_angle + 180) % 360 - 180
        lut = preheated_lut if preheated_lut is not None else get_lut(mode, beta, xr*xr+yr*yr)
        
        out = np.empty((dst_h, dst_w), dtype=np.uint8)
        self.select_kernel(mode)(image, out, src_h, src_w, dst_h, dst_w, sx, sy, dx, dy, lut, lut, xr, yr, float(delta_theta))
        return out if return_uint8 else out.astype(np.float32) / 255.0

_backend = None
def get_backend():
    global _backend
    if _backend is None: _backend = PHProcessor()
    return _backend

def rotate(img, angle_deg, mode='hq', beta=None, reshape=False, return_uint8=False):
    """
    Standard PH rotation entry point. 
    Supported modes: 'fast' (4-tap), 'hq' (8-tap), 'shq' (12-tap).
    Optional: beta (spline order), reshape (auto-expand size).
    """
    return get_backend().rotate(img, angle_deg, mode=mode, beta=beta, reshape=reshape, return_uint8=return_uint8)
