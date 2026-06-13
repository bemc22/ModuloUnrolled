# Deep Lightweight Unrolled Network for High Dynamic Range Modulo Imaging
---

## 📌 Overview

**ModuloUnrolled** addresses the phase unwrapping/modulo reconstruction problem by combining classical physics-based optimization with deep learning. 
* **Formulation**: It leverages an unrolled ADMM scheme representing a Plug-and-Play (PnP) framework. 
* **Denoiser Prior**: Relies on a deep ResUNet block embedded inside the optimization iterations to handle the non-convex proximal mappings of modulo reconstruction.

---

## 📂 Repository Structure

* `config.py` — Holds configurations and structural model size settings.
* `demo.py` — Execution script showcasing inference on custom examples.
* `train_denoiser.py` — Pre-trains the deep denoiser network prior (Stage 1).
* `train_unrolled.py` — Jointly trains the unrolled reconstruction framework end-to-end with equivariant regularization (Stage 2).
* `libs/` — Under-the-hood modules (e.g., U-Net architectures, dataset loaders, ADMM/PnP dynamics).
* `ckpts/` — Default directory storing pre-trained weights (`.pth`).

---

## ⚙️ Workflow & Training Pipeline

Training is split into a robust **two-stage pipeline** to ensure stable convergences:

### Stage 1: Denoiser Pre-training
First, train the deep ResUNet denoiser prior autonomously on simulated noisy patterns:
```bash
python train_denoiser.py
```

### Stage 2: End-to-End Unrolled Finetunning
With the pre-trained denoiser as a stable prior, train the deep unrolled ADMM network (`Unrolled`) end-to-end:
```bash
python train_unrolled.py
```
This stage incorporates **equivariant regularization** (invariant to intensity changes/saturation variations) to enhance generalizability.

---

## 🚀 Running the Demo

Test the reconstruction pipeline on a provided sample input (`example.npy`) and plot the final unwrapping comparisons:
```bash
python demo.py
```