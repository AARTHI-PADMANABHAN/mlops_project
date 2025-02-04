# Linear Regression Models

This repository contains two versions of a linear regression model, designed to progressively improve predictive accuracy by incorporating additional features. The models are versioned to ensure traceability and maintainability.
## Overview of Models

- First Model (model_v1.pkl): Uses X1 as the sole predictor for y, providing a simple baseline.
- Updated Model (model_v2.pkl): Expands on the first model by incorporating X2, aiming to enhance prediction accuracy.

## Project Organization
- Training Scripts:
    - model1.py – Trains the first model using only X1 and saves it as model_v1.pkl.
    - model2.py – Trains the second model with both X1 and X2, saving it as model_v2.pkl.
- Storage:
    - Trained models are saved in the models/ directory.
    - The dataset used for training is organized within the data/ directory.

## Version Control & Comparison
- Baseline Model (X1 Only): Establishes initial predictive performance.
- Enhanced Model (X1 & X2): Introduces an additional feature to potentially lower error and improve accuracy.
- Both models are available in the repository, ensuring previous versions remain accessible for comparison.
