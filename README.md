# Biomass-Prediction-using-SSL-DINO-Feature-Fusion
# CSIRO Grass Biomass Prediction

This repository contains a feature extraction and stacking pipeline for predicting grass biomass from images, developed for the [CSIRO Grass Biomass Prediction competition](https://www.kaggle.com/c/csiro-biomass).

## Overview

The project combines **self-supervised learning (SSL)** and **DINO features** for robust image feature extraction. Features from both models are concatenated and used in a **stacked ensemble of LightGBM, XGBoost, and Ridge regression** to predict multiple biomass targets:

- `Dry_Green_g`
- `Dry_Dead_g`
- `Dry_Clover_g`
- `Dry_Total_g`
- `GDM_g`

The simclr pipeline includes:
- Combines **image embeddings + tabular features** for multi-modal regression.
- Self-supervised pretraining improves feature quality for downstream regression.
- Two-stage training:
  1. SSL backbone training with auxiliary loss.
  2. Regression head training & fine-tuning backbone layers.
- Early stopping and learning rate scheduling ensure stable convergence.


The simclr-dino pipeline includes:
- Multi-seed training for stability
- Test-time augmentation (TTA) for improved predictions
- Post-processing to enforce physical constraints between biomass components



## Key Contributions

- Trained a **custom SSL ResNet18 model** on the training dataset.
- The SSL pipeline alone scored 0.5 R²
- Integrated **DINO features** and fused them with SSL embeddings.
- Built a **stacked regression ensemble** with LightGBM, XGBoost, and Ridge.
- Implemented post-processing to enforce biomass sum constraints.
- Achieved a **weighted R² of 0.53** on the competition validation set.

## Acknowledgments

- The stacking approach was **adapted from the public kaggle notebook by zjayzZ** 
