# Hybrid CNN-GAT Fusion for Facial Emotion Recognition

Code and experiment records for the paper:

> M. J. I. Rana, R. A. Druba, I. A. Oruzgani, **"Hybrid CNN-GAT Fusion for Facial Emotion Recognition Using Appearance and Landmark Geometry,"**
> *2026 IEEE 3rd International Student Conference on Digital Generation*, Astana, Kazakhstan, 21–23 October 2026.

The model fuses two branches:

- **Appearance:** a pretrained ResNet-18 (layer1 and layer2 frozen), followed by a 1×1 conv to 256 channels, BN, ReLU and global average pooling. Output: a 256-d vector.
- **Geometry:** 97 MediaPipe Face Landmarker points with (x, y, z) node features, connected as a k-NN graph (k = 6). A 2-layer GAT and global mean pooling produce a 48-d vector.
- **Fusion:** the two vectors are concatenated (304-d), then passed through FC-512, BN, ReLU, Dropout 0.7 and a linear classifier.

```
Input image ──► ResNet-18 ──► 256-d ─┐
     │                               ├─► concat 304-d ─► FC 512 + BN + ReLU + Dropout 0.7 ─► emotion
     └─► MediaPipe 97 pts ─► k-NN (k=6) ─► 2-layer GAT ─► 48-d ─┘
```

## Results reported in the paper

| Dataset | Accuracy | Precision | Recall | Weighted F1 | Macro F1 | Test images |
|---|---|---|---|---|---|---|
| AffectNet (8 classes) | 69.98 | 70.60 | 69.98 | 70.14 | 69 | 5,577 |
| FER+ (8 classes) | 80.46 | 80.34 | 80.46 | 80.17 | 66 | 3,573 |
| OAHEGA (6 classes) | 82.99 | 83.04 | 82.99 | 82.96 | 84.22 | 3,046 |

Ablation on FER+ (Table VII):

| Model | Accuracy | Weighted F1 |
|---|---|---|
| CNN only | 69.20 | 68.00 |
| GAT only | 34.90 | 26.30 |
| CNN + GAT (proposed) | **80.46** | **80.17** |

The original logs and figures behind every number are in [`results/`](results/README.md).

## Which notebook produces which result

| Paper item | Notebook | Setting |
|---|---|---|
| Table II, Fig. 2 (AffectNet) | [`notebooks/01_cnn_gat_fusion.ipynb`](notebooks/01_cnn_gat_fusion.ipynb) | `DATASET_NAME = "AffectNet"` |
| Table III, Fig. 3 (FER+) | [`notebooks/01_cnn_gat_fusion.ipynb`](notebooks/01_cnn_gat_fusion.ipynb) | `DATASET_NAME = "FER+"` |
| Table IV, Fig. 4 (OAHEGA) | [`notebooks/01_cnn_gat_fusion.ipynb`](notebooks/01_cnn_gat_fusion.ipynb) | `DATASET_NAME = "OAHEGA"` |
| Table V (summary) | same three runs | — |
| Table VII, CNN only | [`notebooks/02_ablation_cnn_only.ipynb`](notebooks/02_ablation_cnn_only.ipynb) | FER+ |
| Table VII, GAT only | [`notebooks/03_ablation_gat_only.ipynb`](notebooks/03_ablation_gat_only.ipynb) | FER+ |

`notebooks/archive/` holds the original notebooks exactly as they were run, with their outputs. The notebooks in `notebooks/` are cleaned copies of them. The only changes are:
- relative paths instead of local Windows paths;
- fixed random seeds;
- cells ordered so they run top to bottom;
- separate checkpoint names for each experiment.

## Setup

Tested with Python 3.12, CUDA 12.4 and an NVIDIA GPU.

```bash
git clone <this-repo-url>
cd Hybrid-CNN-GAT-FER
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The MediaPipe landmark model is already included at `models/face_landmarker.task`. It is Google's `face_landmarker` float16 v1 model, licensed under Apache-2.0.

## Data

The datasets are **not** included; download them from Kaggle. See [`data/README.md`](data/README.md) for the links and the exact folder layout.

## Running

```bash
cd notebooks
jupyter notebook
```

1. Open `01_cnn_gat_fusion.ipynb`.
2. Set `DATASET_NAME` (`"AffectNet"`, `"FER+"` or `"OAHEGA"`).
3. Run all cells. The best checkpoint is saved to `checkpoints/cnn_gat_fusion_<DATASET>.pth`, then evaluated.

Repeat steps 2 and 3 for each dataset. Run `02_` and `03_` for the ablation study.

Landmark extraction runs MediaPipe on the CPU for every image in every epoch. Expect long epochs, for example several hours for AffectNet.

## Reproducibility notes

Read these before comparing your numbers with the paper.

1. **The original runs were not seeded.** The cleaned notebooks fix the seeds, but GPU non-determinism, AMP and the random landmark noise still cause small run-to-run differences. Expect results close to the reported ones, not identical to the last digit.
2. **AffectNet and FER+ test-time landmarks.** In the original evaluation runs, the official test set was built with the dataset-class defaults: all 478 landmarks and k = 10. Training used 97 landmarks and k = 6. `EVAL_LIKE_ORIGINAL_RUN = True`, the default in `01_cnn_gat_fusion.ipynb`, reproduces that original call. Set it to `False` to evaluate with the 97-landmark, k = 6 setting described in the paper.
3. **Shared transform.** The train, val and held-out subsets share one underlying dataset object. The last transform assigned (`transforms_val`) is therefore also used during training, and landmark-noise augmentation (`augment_landmarks=True`) is applied to all three subsets. This is kept as-is because it is how the reported numbers were produced.
4. **Ablation evaluation sets.**
   - CNN-only (69.20%) was measured on the FER+ `validation` folder (8,341 images).
   - GAT-only (34.90%) was measured on a 20% split of the FER+ `train` folder (13,276 images).
   - The fusion model was measured on the FER+ `test` folder.
   - The GAT-only log gives a weighted F1 of 26.9% and a macro F1 of 26.3%. Table VII reports 26.30.
5. **No fusion weights are shipped.** The trained checkpoints for the paper configuration were not kept, so retrain with the notebooks.

## Repository layout

```
├── notebooks/
│   ├── 01_cnn_gat_fusion.ipynb      # main model (Tables II–V, Figs. 2–4)
│   ├── 02_ablation_cnn_only.ipynb   # Table VII, CNN only
│   ├── 03_ablation_gat_only.ipynb   # Table VII, GAT only
│   └── archive/                     # original notebooks with their outputs
├── models/face_landmarker.task      # MediaPipe Face Landmarker
├── data/README.md                   # dataset download + folder layout
├── results/                         # original logs, confusion matrices, ablation history
├── requirements.txt
└── CITATION.cff
```

## License

The code is released under the [MIT License](LICENSE). The bundled `models/face_landmarker.task` is Google's MediaPipe model (Apache-2.0). The datasets are not included and keep their own terms.

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)).
