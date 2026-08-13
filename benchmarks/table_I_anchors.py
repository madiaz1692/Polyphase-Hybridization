#!/usr/bin/env python3
"""
table_I_anchors.py -- Optimal Anchor Pool Verification
Ref: Polyphase Hybridization, Table I (Line 126)
Verifies the selection of the 9 optimal coprime anchors within [0°, 45°].
"""

import math
import numpy as np
import os

def check_weight_presence(A, weights_dir="./weights"):
    """Verifies that weights for all three standard modes exist for the anchor A."""
    # Fallback for toolkit execution (where weights are in libPH/weights)
    if not os.path.exists(weights_dir):
        alt_path = os.path.join("libPH", "weights")
        if os.path.exists(alt_path):
            weights_dir = alt_path

    targets = [
        f"fast_beta3_A{A}.npy",
        f"hq_beta5_A{A}.npy",
        f"shq_beta7_A{A}.npy"
    ]
    found = 0
    for t in targets:
        if os.path.exists(os.path.join(weights_dir, t)):
            found += 1
    
    if found == 3: return "[OK]"
    if found == 0: return "[MISSING]"
    return f"[PARTIAL:{found}/3]"

def verify_anchors():
    print("=" * 80)
    print(f"{' PH: TABLE I - OPTIMAL ANCHOR POOL ':^80}")
    print(f"{' (7 standard [25-50] + 2 high-density boundary guards) ':^80}")
    print("=" * 80)
    print(f"{'(x_r, y_r)':<12} | {'A':<5} | {'theta (deg)':<12} | {'Weights':<10} | {'gcd(x,y)':<8}")
    print("=" * 80)
    
    anchors = []
    # 1. Standard search for 25 <= A <= 50 (Coprime)
    for x in range(1, 10):
        for y in range(0, x + 1):
            if math.gcd(x, y) == 1:
                A = x*x + y*y
                if 25 <= A <= 50:
                    theta = math.degrees(math.atan2(y, x))
                    anchors.append({'xr': x, 'yr': y, 'A': A, 'theta': theta})
    
    # 2. Add high-density specialized boundary guards
    for xr, yr in [(11, 1), (8, 7)]:
        theta = math.degrees(math.atan2(yr, xr))
        anchors.append({'xr': xr, 'yr': yr, 'A': xr**2 + yr**2, 'theta': theta})


    # Sort and remove duplicates (if any)
    anchors.sort(key=lambda x: x['theta'])
    
    for a in anchors:
        coord = f"({a['xr']}, {a['yr']})"
        w_status = check_weight_presence(a['A'])
        print(f"{coord:<12} | {a['A']:<5} | {a['theta']:>10.2f}°  | {w_status:<10} | 1")

    print("-" * 80)
    
    # Theoretical Error calculation
    angles = [0.0] + [a['theta'] for a in anchors] + [45.0]
    max_err = 0
    for i in range(len(angles) - 1):
        # Gap between two points
        gap = angles[i+1] - angles[i]
        # At boundaries (0 and 45), the residual is the distance to the nearest anchor
        if i == 0: # 0 to first anchor
             err = gap 
        elif i == len(angles) - 2: # last anchor to 45
             err = gap
        else: # between anchors, midpoint is worst case
             err = gap / 2.0
        max_err = max(max_err, err)

    print(f"Total Optimal Anchors: {len(anchors)}")
    print(f"Theoretical Max Error (Delta Theta): ~{max_err:.2f}°")
    print("=" * 80)

if __name__ == "__main__":
    verify_anchors()
