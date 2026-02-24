# PDU: Proactive Domain Unification for Robust Echocardiography Segmentation


![pipeline](pud_image/pipeline.jpg)

This repository provides an implementation of a **domain-unified pipeline** for ultrasound image segmentation, including:

- **Stage I**: Structure-constrained domain unifier training (ControlNet diffusion)
- **Stage II**: Inference-time domain unification
- **Stage III**: Representation learning (BYOL hypersphere)
- **Stage IV**: Quality scoring & reliability fusion (frequency-domain fusion)

> Put the main figure at: `pud_image/pipeline.jpg` (you said you will upload it there).

---

## Repository Structure (recommended)

```text
.
├─ DomainUnifiedSegmentation/        # main code (your implementation)
├─ third_party/
│  ├─ controlnet/                    # from Stage I/II (keep original LICENSE)
│  └─ byol_pytorch/                  # from Stage III (keep original LICENSE)
├─ scripts/                          # runnable wrappers (optional)
├─ configs/                          # json/yaml configs
├─ pud_image/
│  └─ pipeline.jpg                   # main figure
├─ checkpoints/                      # (gitignored) model checkpoints
├─ outputs/                          # (gitignored) inference outputs
├─ requirements.txt
├─ LICENSE
└─ README.md
````

If you keep the original folders like `Stage1and2/` and `Stage3/`, just replace `third_party/controlnet` and `third_party/byol_pytorch` in the commands below with your actual paths.

---

## 1. Environment

### Option A) Conda (recommended)

```bash
conda create -n pdu python=3.10 -y
conda activate pdu
pip install -r requirements.txt
```

### Option B) Start from the ControlNet environment

If the ControlNet part provides `environment.yaml`, you can use it; otherwise stick with `requirements.txt`.

Example:

```bash
conda env create -f third_party/controlnet/environment.yaml
conda activate controlnet
pip install -r requirements.txt
```

---

## 2. Data Preparation

### 2.1 Recommended folder layout

```text
data/
  source/
    images/            # source domain images
    masks/             # source domain segmentation masks (binary or class-index)
  target/
    images/            # target domain images (unlabeled)
  edges/               # optional: edge GT or predicted edges
  unified/             # outputs of domain unifier (Stage II)
  fused/               # outputs after reliability fusion (Stage IV)
```

### 2.2 Notes

* Supported image extensions: `.png`, `.jpg`, `.jpeg`.
* If you do not provide edge GT, edges can be derived from masks or predicted by an edge detector.

---

## 3. Stage I — Domain Unifier Training (ControlNet)

### 3.1 (Optional) Train an edge detector (HED-like)

Train edge network (probability-output):

```bash
python DomainUnifiedSegmentation/train_edge.py \
  --images_dir data/source/images \
  --masks_dir  data/source/masks \
  --out_ckpt   checkpoints/hed_lv_prob.pth \
  --epochs 30 --batch_size 8 --lr 1e-4 --resize 256 --device cuda
```

Export predicted edges (for ControlNet hints):

```bash
python DomainUnifiedSegmentation/export_edges.py \
  --images_dir data/source/images \
  --out_dir    data/edges/source_pred \
  --ckpt       checkpoints/hed_lv_prob.pth \
  --resize     512 --device cuda
```

### 3.2 Prepare ControlNet JSON

Without edge hints:

```bash
python DomainUnifiedSegmentation/prepare_controlnet_json.py \
  --images_dir data/source/images \
  --out_json   configs/train_controlnet_source.json \
  --prompt "Ultrasound"
```

With edge hints:

```bash
python DomainUnifiedSegmentation/prepare_controlnet_json.py \
  --images_dir data/source/images \
  --hints_dir  data/edges/source_pred \
  --out_json   configs/train_controlnet_source.json \
  --prompt "Ultrasound"
```

### 3.3 Train the structure-guided domain unifier

This repo reuses ControlNet code (see **Acknowledgements**). Example:

```bash
python DomainUnifiedSegmentation/train_controlnet_unifier.py \
  --stage1_root third_party/controlnet \
  --train_json  configs/train_controlnet_source.json \
  --sd_ckpt     /path/to/stable-diffusion-v1-5.ckpt \
  --controlnet_ckpt /path/to/control_sd15_canny.pth \
  --out_dir     checkpoints/controlnet_unifier \
  --max_epochs  50 --batch_size 4 --gpus 1
```

> Adjust `--stage1_root` to your actual folder if you keep `Stage1and2/` instead of `third_party/controlnet/`.

---

## 4. Stage II — Inference-time Domain Unification

Run unifier on target domain images and save unified-domain outputs:

```bash
python DomainUnifiedSegmentation/infer_unify_only.py \
  --stage1_root third_party/controlnet \
  --finetuned_ckpt checkpoints/controlnet_unifier/last.ckpt \
  --input_dir  data/target/images \
  --output_dir data/unified/target \
  --prompt "Ultrasound" --device cuda
```

Batch unify (all splits / folders if you have):

```bash
python DomainUnifiedSegmentation/infer_unifier_all_splits_v2.py \
  --stage1_root third_party/controlnet \
  --finetuned_ckpt checkpoints/controlnet_unifier/last.ckpt \
  --root_dir data \
  --prompt "Ultrasound" --device cuda
```

---

## 5. Stage III — Representation Learning (BYOL hypersphere)

Train BYOL-based representation model using **target images** and **unified images**:

```bash
python Stage3/train_byol_hypersphere.py \
  --target_dir data/target/images \
  --unified_dir data/unified/target \
  --out_ckpt checkpoints/byol_hypersphere.ckpt \
  --image_size 512 --batch_size 32 --max_epochs 200 --device cuda
```

Notes:

* BYOL implementation lives in `third_party/byol_pytorch/` (see **Acknowledgements**).
* If your code uses a different entry script, replace the command accordingly.

---

## 6. Stage IV — Quality Scoring & Reliability Fusion

Compute quality score `S` and fuse frequency components (DWT/IDWT-based):

```bash
python Stage3/compute_S_and_fuse_calib.py \
  --target_dir data/target/images \
  --unified_dir data/unified/target \
  --repr_ckpt  checkpoints/byol_hypersphere.ckpt \
  --out_dir    data/fused/target \
  --device cuda
```

The fused images in `data/fused/target/` can be used as the final “reliable unified domain” inputs.

---

## 7. Downstream Segmentation Training & Inference

### 7.1 Train segmentation model on source domain

```bash
python DomainUnifiedSegmentation/train_seg.py \
  --images_dir data/source/images \
  --masks_dir  data/source/masks \
  --out_ckpt   checkpoints/seg_unet.pth \
  --num_classes 2 --epochs 100 --batch_size 8 --device cuda
```

### 7.2 Inference with domain unification (target domain)

```bash
python DomainUnifiedSegmentation/infer_seg_unify.py \
  --stage1_root third_party/controlnet \
  --finetuned_ckpt checkpoints/controlnet_unifier/last.ckpt \
  --seg_ckpt checkpoints/seg_unet.pth \
  --input_dir data/target/images \
  --output_dir outputs/seg_target \
  --prompt "Ultrasound" --device cuda
```

---

## 8. Acknowledgements

This project reuses and adapts components from the following work:

* **ADAptation: Reconstruction-based Unsupervised Active Learning for Breast Ultrasound Diagnosis**
  Codebase: [https://github.com/miccai25-966/ADAptation](https://github.com/miccai25-966/ADAptation)

We thank the authors for releasing their implementation, which is helpful for building our diffusion/ControlNet training pipeline and related utilities.

---

## 9. License

* Code in this repository: see `LICENSE`.
* Third-party code under `third_party/`: please follow the original licenses included in those folders.

---

## 10. Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{YOUR_PAPER_KEY,
  title     = {YOUR_TITLE},
  author    = {YOUR_AUTHORS},
  booktitle = {MICCAI},
  year      = {2025}
}
```

```
```
