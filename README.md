# PDU: Proactive Domain Unification for Robust Echocardiography Segmentation


![pipeline](image/miccaiPUD.jpg)

This repository provides an implementation of a **domain-unified pipeline** for ultrasound image segmentation, including:

- **Stage I**: Structure-constrained domain unifier training (ControlNet diffusion)
- **Stage II**: Inference-time domain unification
- **Stage III**: Representation learning (BYOL hypersphere)
- **Stage IV**: Quality scoring & reliability fusion (frequency-domain fusion)


---


## Stage I — Domain Unifier Training (ControlNet)

### (Optional) Train an edge detector (HED-like)

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

### Prepare ControlNet JSON

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

### Train the structure-guided domain unifier

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

## Stage II — Inference-time Domain Unification

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

## Stage III — Representation Learning (BYOL hypersphere)

Train BYOL-based representation model using **target images** and **unified images**:

```bash
python Stage3/train_byol_hypersphere.py \
  --target_dir data/target/images \
  --unified_dir data/unified/target \
  --out_ckpt checkpoints/byol_hypersphere.ckpt \
  --image_size 512 --batch_size 32 --max_epochs 200 --device cuda
```
---

## Stage IV — Quality Scoring & Reliability Fusion

Compute quality score `S` and fuse frequency components (DWT/IDWT-based):

```bash
python Stage3/compute_S_and_fuse_calib.py \
  --target_dir data/target/images \
  --unified_dir data/unified/target \
  --repr_ckpt  checkpoints/byol_hypersphere.ckpt \
  --out_dir    data/fused/target \
  --device cuda



---
```bash
License

Code in this repository: see LICENSE.

Third-party code under third_party/: please follow the original licenses included in those folders.
---

