#!/usr/bin/env python3
"""
table_I_anchors.py -- Optimal Anchor Pool Verification
Ref: Polyphase Hybridization (Array Journal), Table 1
Verifies the selection of the 11 optimal coprime anchors within [0°, 45°]
(9 standard in 25 <= A <= 60 plus 2 high-density boundary guards).
"""

import math
import os

def check_weight_presence(A, weights_dir=None):
    """Verifies that weights for all three standard modes exist for the anchor A."""
    candidates = [
        "../weights",
        "./weights",
        "./libPH/weights",
        os.path.join(os.path.dirname(__file__), "..", "libPH", "weights"),
        os.path.join(os.path.dirname(__file__), "..", "weights")
    ]
    actual_dir = None
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            actual_dir = c
            break
            
    if actual_dir is None: return "[MISSING_DIR]"

    targets = [
        f"fast_beta3_A{A}.npy",
        f"hq_beta5_A{A}.npy",
        f"shq_beta7_A{A}.npy"
    ]
    found = 0
    for t in targets:
        if os.path.exists(os.path.join(actual_dir, t)):
            found += 1
    
    if found == 3: return "[OK]"
    if found == 0: return "[MISSING]"
    return f"[PARTIAL:{found}/3]"

def verify_anchors():
    print("=" * 85)
    print(f"{' PH: TABLE 1 - OPTIMAL 11-ANCHOR POOL ':^85}")
    print(f"{' (9 standard [25 <= A <= 60] + 2 boundary guards, all gcd(x,y)=1) ':^85}")
    print("=" * 85)
    print(f"{'(x_r, y_r)':<12} | {'theta_A (deg)':<14} | {'A':<5} | {'|DeltaTheta_max|':<16} | {'LUT (KiB)':<10} | {'Weights':<8}")
    print("-" * 85)
    
    MAIN_POOL = [
        (11, 1), (7, 1), (6, 1), (5, 1), (7, 2), 
        (5, 2), (7, 3), (5, 3), (4, 3), (5, 4), (8, 7)
    ]
    
    # Calculate residual bounds matching Table 1
    anchors = []
    for x, y in MAIN_POOL:
        theta = math.degrees(math.atan2(y, x))
        A = x*x + y*y
        lut_kib = (A * 8 * 2 * 4) / 1024.0 # D=8 (HQ baseline)
        anchors.append({'xr': x, 'yr': y, 'A': A, 'theta': theta, 'lut': lut_kib})

    # Sort by angle
    anchors.sort(key=lambda x: x['theta'])
    
    # Compute per-anchor max residual in pool
    thetas = [a['theta'] for a in anchors]
    for i, a in enumerate(anchors):
        if i == 0:
            # (11,1) boundary guard: from 0 deg to midpoint with next
            delta_max = a['theta'] # at 0 deg
        elif i == len(anchors) - 1:
            # (8,7) boundary guard: from midpoint to 45 deg
            delta_max = 45.0 - a['theta'] # at 45 deg
        else:
            left_half = (a['theta'] - thetas[i-1]) / 2.0
            right_half = (thetas[i+1] - a['theta']) / 2.0
            delta_max = max(left_half, right_half)
        a['delta_max'] = delta_max

    total_lut = 0.0
    for a in anchors:
        coord = f"({a['xr']}, {a['yr']})"
        w_status = check_weight_presence(a['A'])
        total_lut += a['lut']
        print(f"{coord:<12} | {a['theta']:>12.2f}° | {a['A']:<5} | {a['delta_max']:>14.2f}° | {a['lut']:>8.1f}   | {w_status:<8}")

    print("-" * 85)
    print(f"{'Total (all 11 anchors):':<52} | {total_lut:>8.1f} KiB |")
    print("=" * 85)

if __name__ == "__main__":
    verify_anchors()
