# Datasets

The images are not part of this repository. Download them from the Kaggle pages cited in the paper and place them here with **exactly these folder names**. The code detects the dataset from the folder name.

| Folder | Source |
|---|---|
| `data/AffectNet/` | https://www.kaggle.com/datasets/mstjebashazida/affectnet |
| `data/FER+/` | https://www.kaggle.com/datasets/arnabkumarroy02/ferplus |
| `data/OAHEGA (Facial Emotion Recognition)/` | https://www.kaggle.com/datasets/alexa1979/oahega-emotion |

## Expected layout and image counts (as used in the paper)

```
data/
├── AffectNet/
│   ├── train/{anger,contempt,disgust,fear,happy,neutral,sad,surprise}/   22,288 images
│   └── test/{anger,contempt,disgust,fear,happy,neutral,sad,surprise}/     5,577 images
├── FER+/
│   ├── train/{angry,contempt,disgust,fear,happy,neutral,sad,surprise}/   66,379 images
│   ├── validation/{...same 8...}/                                         8,341 images
│   └── test/{...same 8...}/                                               3,573 images
└── OAHEGA (Facial Emotion Recognition)/
    └── {Ahegao,Angry,Happy,Neutral,Sad,Surprise}/                        15,232 images used
```

**Image extensions:** the loader collects `*.jpg` and `*.png` with `glob`. On Windows, where the paper's runs were done, this matching is case-insensitive, so `.JPG` and `.PNG` files were included, while `.jpeg` and `.bmp` files were skipped. That is how the OAHEGA folder (15,453 files) became the 15,232 images used:

| Class | Images used |
|---|---|
| Ahegao | 1205 |
| Angry | 1303 |
| Happy | 3673 |
| Neutral | 3945 |
| Sad | 3875 |
| Surprise | 1231 |

On Linux or macOS, `glob` is case-sensitive. To get the same image set there, rename the `.JPG` and `.PNG` files to lowercase extensions first.

Per-class test counts used for checking your copy:

- AffectNet `test`: anger 720, contempt 637, disgust 688, fear 609, happy 857, neutral 569, sad 598, surprise 899
- FER+ `test`: angry 322, contempt 30, disgust 21, fear 98, happy 929, neutral 1274, sad 449, surprise 450

How each dataset is split:
- **FER+:** the class folder is called `angry`; the code maps it to the `anger` label.
- **OAHEGA:** has no official split. The notebook makes a seeded (42) 64 / 16 / 20 train / val / test split.
- **AffectNet and FER+:** the `train` folder is split the same way for training and validation. The official `test` folder is used for the reported results.
