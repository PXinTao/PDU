
# PDU: Proactive Domain Unification for Robust Echocardiography Segmentation

Official implementation of **Proactive Domain Unification (PDU)** for robust cross-center echocardiography segmentation.

> **MICCAI 2026 Early Accept**   


![pipeline](Image/miccaiPUD.jpg)

## Overview

PDU is an inference-time input-domain unification framework for robust echocardiography segmentation under cross-center and cross-vendor appearance shifts.

Instead of requiring target-domain labels or updating the segmenter at test time, PDU maps heterogeneous inputs into a fixed source-aligned appearance space before segmentation. The segmentation model is trained on PDU-unified source images and PDU is applied consistently at inference.

The pipeline contains four main stages:

- **Stage I**: Structure-constrained domain unifier training with ControlNet diffusion
- **Stage II**: Source/target domain unification using the trained unifier
- **Stage III**: Reliability representation learning with BYOL hypersphere embeddings
- **Stage IV**: Reliability-guided frequency fusion with DWT/IDWT

---

## Repository Structure

```text
PDU/
├── DomainUnifiedSegmentation/
│   ├── train_edge.py
│   ├── export_edges.py
│   ├── prepare_controlnet_json.py
│   ├── train_controlnet_unifier.py
│   ├── infer_unify_only.py
│   └── infer_unifier_all_splits_v2.py
├── Stage3/
│   ├── train_byol_hypersphere.py
│   └── compute_S_and_fuse_calib.py
├── third_party/
│   └── controlnet/
├── Image/
│   └── miccaiPUD.jpg
├── checkpoints/
├── configs/
├── data/
├── LICENSE
└── README.md
````

---

## Installation

Create a Python environment and install the required packages.

```bash
conda create -n pdu python=3.9 -y
conda activate pdu
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

Install other dependencies:

```bash
pip install numpy opencv-python pillow tqdm scikit-image scikit-learn matplotlib
pip install einops pytorch-lightning
```

For ControlNet-related training and inference, please also install the dependencies required by the ControlNet code under `third_party/controlnet/`.

---

## Data Preparation

A recommended data layout is:

```text
data/
├── source/
│   ├── images/
│   └── masks/
├── target/
│   ├── images/
│   └── masks/          # optional, only for evaluation
├── edges/
│   └── source_pred/
├── unified/
│   ├── source/
│   └── target/
└── fused/
    └── target/
```

Here, `source` denotes the labeled training domain used to train the segmentation model, and `target` denotes the deployment domain.

---

## Stage I — Structure-Constrained Domain Unifier Training

### Optional: Train an LV Edge Detector

Train an HED-like edge detector using source-domain images and masks.

```bash
python DomainUnifiedSegmentation/train_edge.py \
  --images_dir data/source/images \
  --masks_dir  data/source/masks \
  --out_ckpt   checkpoints/hed_lv_prob.pth \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --resize 256 \
  --device cuda
```

Export predicted LV boundary cues as ControlNet hints.

```bash
python DomainUnifiedSegmentation/export_edges.py \
  --images_dir data/source/images \
  --out_dir    data/edges/source_pred \
  --ckpt       checkpoints/hed_lv_prob.pth \
  --resize     512 \
  --device cuda
```

### Prepare ControlNet Training JSON

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

### Train the Domain Unifier

This repository reuses ControlNet code. Please make sure `--stage1_root` points to your local ControlNet directory.

```bash
python DomainUnifiedSegmentation/train_controlnet_unifier.py \
  --stage1_root third_party/controlnet \
  --train_json  configs/train_controlnet_source.json \
  --sd_ckpt     /path/to/stable-diffusion-v1-5.ckpt \
  --controlnet_ckpt /path/to/control_sd15_canny.pth \
  --out_dir     checkpoints/controlnet_unifier \
  --max_epochs  50 \
  --batch_size 4 \
  --gpus 1
```

If your ControlNet code is stored under another folder, replace `third_party/controlnet` with the corresponding path.

---

## Stage II — Domain Unification

After Stage I, use the trained unifier to generate source-aligned images.

### Unify Target Images

```bash
python DomainUnifiedSegmentation/infer_unify_only.py \
  --stage1_root third_party/controlnet \
  --finetuned_ckpt checkpoints/controlnet_unifier/last.ckpt \
  --input_dir  data/target/images \
  --output_dir data/unified/target \
  --prompt "Ultrasound" \
  --device cuda
```

### Unify Source Images

For consistency with the paper setting, the segmentation model should be trained on PDU-unified source images.

```bash
python DomainUnifiedSegmentation/infer_unify_only.py \
  --stage1_root third_party/controlnet \
  --finetuned_ckpt checkpoints/controlnet_unifier/last.ckpt \
  --input_dir  data/source/images \
  --output_dir data/unified/source \
  --prompt "Ultrasound" \
  --device cuda
```

### Batch Unification

If your data are organized into multiple folders or splits:

```bash
python DomainUnifiedSegmentation/infer_unifier_all_splits_v2.py \
  --stage1_root third_party/controlnet \
  --finetuned_ckpt checkpoints/controlnet_unifier/last.ckpt \
  --root_dir data \
  --prompt "Ultrasound" \
  --device cuda
```

---

## Stage III — Reliability Representation Learning

PDU learns a reliability embedding from **source raw--unified pairs**. This avoids requiring target-domain labels and keeps the reliability model anchored to the source-domain training distribution.

Train the BYOL hypersphere representation model:

```bash
python Stage3/train_byol_hypersphere.py \
  --target_dir data/source/images \
  --unified_dir data/unified/source \
  --out_ckpt checkpoints/byol_hypersphere.ckpt \
  --image_size 512 \
  --batch_size 32 \
  --max_epochs 200 \
  --device cuda
```

> Note: The argument name `--target_dir` follows the original script interface. In the PDU training protocol, this directory should point to the **raw source images**, while `--unified_dir` should point to the corresponding **PDU-unified source images**.

---

## Stage IV — Reliability-Guided Frequency Fusion

At inference time, PDU computes a reliability score and fuses raw and unified target images in the frequency domain. The low-frequency anatomical structure is inherited from the raw image, while high-frequency source-style cues are injected under reliability control.

```bash
python Stage3/compute_S_and_fuse_calib.py \
  --target_dir data/target/images \
  --unified_dir data/unified/target \
  --repr_ckpt  checkpoints/byol_hypersphere.ckpt \
  --out_dir    data/fused/target \
  --device cuda
```

The fused images will be saved to:

```text
data/fused/target/
```

These fused images can then be used as the final inputs to the frozen segmentation model.

---

## Segmentation Training and Evaluation

In the paper setting, segmentation models are trained on the PDU-unified source domain and evaluated on PDU-processed target images.

A typical protocol is:

```text
Train segmentation model:
    input  = data/unified/source/
    label  = data/source/masks/

Inference:
    input  = data/fused/target/
    model  = frozen segmentation model
```

For the raw baseline, train the same segmentation architecture on:

```text
input = data/source/images/
label = data/source/masks/
```

and evaluate directly on:

```text
input = data/target/images/
```

This ensures a clear comparison between direct cross-domain deployment and proactive input-domain unification.

---

## Important Notes

* PDU does **not** require target-domain labels.
* PDU does **not** update the segmentation model at test time.
* PDU is a train/test input-domain redesign: the segmenter is trained on PDU-unified source images and PDU is applied consistently at inference.
* The reliability encoder is trained using source raw--unified pairs.
* Target images are processed only at inference for unification, reliability estimation, and frequency fusion.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{pang2026pdu,
  title     = {Proactive Domain Unification for Robust Echocardiography Segmentation},
  author    = {Pang, Xintao and Yang, Jinlin and Sun, Yue and Gao, Zhifan and Li, Wei and Tan, Tao},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026}
}
```

---

## Acknowledgements

This repository builds upon and reuses components from the following projects:

- [ControlNet](https://github.com/lllyasviel/ControlNet)
- Stable Diffusion
- [ADAptation](https://github.com/miccai25-966/ADAptation), the official codebase for *ADAptation: Reconstruction-based Unsupervised Active Learning for Breast Ultrasound Diagnosis*, from which we adapted parts of the diffusion-based reconstruction/unification and hypersphere representation pipeline.

We thank the authors of these projects for making their code and resources publicly available. Please follow the original licenses of third-party code included under `third_party/`.

---

## License

Code in this repository is released under the license specified in `LICENSE`.

Third-party code under `third_party/` is governed by the original licenses of the corresponding projects.


