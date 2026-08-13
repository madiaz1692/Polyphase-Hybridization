# Polyphase Hybridization (PH)
### Experimental Validation Suite

This toolkit provides a self-contained, precompiled version of the Polyphase Hybridization (PH) framework for image rotation. It is designed to allow researchers to replicate the empirical results presented in the manuscript: *"Polyphase Hybridization: Decoupling Spline Fidelity from Execution Cost in Image Rotation"*.

**Authors:**
- Maria S. Diaz-Gonzalez
- Jose L. Gonzalez-Mora

---

## 1. Structure
- `replicate_performance.py`: **Master Orchestrator**. Runs the 5 core experiments sequentially.
- `libPH/`: Core framework engine and optimized binaries.
  - `core/`: Optimized SIMD binaries (`.dll` for Windows, `.so` for Linux).
  - `weights/`: Precomputed rational kernels (Farey anchors $A \in [25, 50]$).
- `benchmarks/`: Individual experiment scripts for Tables I to V.

## 2. Requirements
To replicate the results, the following Python libraries are required:
- **NumPy**: Linear algebra and array management.
- **OpenCV**: Baseline comparison for classical kernels.
- **SciPy**: Baseline comparison for recursive IIR spline solvers.
- **Scikit-Image**: Metrics (SSIM/PSNR) and natural image dataset.

To install all dependencies:
```bash
pip install numpy opencv-python scipy scikit-image
```

*Note: All images and phantoms used in the paper are either mathematically generated (Table IV) or retrieved automatically via `skimage.data` (Table III), so no external image assets are required.*

## 3. Quick Start: Full Replication
To replicate the entire experimental suite (Tables I, II, III, IV, and V) from the manuscript, simply run:

```bash
python replicate_performance.py
```

This will sequentially execute the benchmarks for:
1. **Table I**: Optimal Anchor Pool distribution.
2. **Table II**: Computational Scalability (Latency vs OpenCV/SciPy).
3. **Table III**: Harmonized Native Fidelity (PSNR/SSIM).
4. **Table IV**: Analytical Fidelity Suite (Phantoms).
5. **Table V**: Iterative Robustness (360° Round-trips).

## 4. Hardware Support
The provided binaries include:
- **Windows**: x86_64 with AVX2/FMA3 support.
- **Linux**: x86_64 with AVX2/FMA3 support.
- **ARM64**: SVE-optimized binaries (to be tested on AArch64).

The framework includes a bit-accurate scalar fallback that is automatically activated if specific SIMD features are not available on the execution host.
### 5. Visual Debugging
Validation scripts for Table IV and Table V include a visual debugging flag. To inspect the structural stability of the rotation per angle or per step, you can enable this in the script headers:
- `benchmarks/table_IV_stability.py`: Set `SAVE_DEBUG_IMGS = True`
- `benchmarks/table_V_iterative.py`: Set `SAVE_DEBUG_ITERATIVE = True`

This will generate a `debug_*/` directory with lossless PNG exports for visual inspection.

---
© 2026 PH Development Team. Open-source release.
