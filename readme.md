# DA6401 Assignment 2 — Visual Perception Pipeline

A complete multi-task visual perception pipeline on the Oxford-IIIT Pet Dataset
implementing classification, object localization, and semantic segmentation.

## Results
| Task | Metric | Score |
|------|--------|-------|
| Classification | Macro F1 | 0.82 |
| Localization | Acc@IoU=0.75 | 80% |
| Segmentation | Dice Score | 0.856 |

## Links
- **W&B Report:** [INSERT YOUR PUBLIC WANDB REPORT LINK HERE]
- **GitHub:** https://github.com/shadab007-byte/da6401_assignment_2

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
# Task 1 - Classification
python train.py --data_root data/oxford-iiit-pet --skip_loc --skip_seg

# Task 2 - Localization  
python train.py --data_root data/oxford-iiit-pet --skip_cls --skip_seg

# Task 3 - Segmentation
python train.py --data_root data/oxford-iiit-pet --skip_cls --skip_loc
```

## Inference
```bash
python inference.py --data_root data/oxford-iiit-pet
```

## Project Structure
```
├── checkpoints/     # Model checkpoints (downloaded via gdown)
├── data/            # Dataset loader
├── losses/          # Custom IoU loss
├── models/          # VGG11, classifier, localizer, segmentation, multitask
├── train.py         # Training script
├── inference.py     # Evaluation script
└── requirements.txt
```
