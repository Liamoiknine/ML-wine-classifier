# Wine Quality Classification  

This project compares three machine learning models — an Artificial Neural Network, a Random Forest, and a K-Nearest Neighbors classifier — to predict wine quality ratings using a red wine dataset.

## Overview  

The goal was to predict wine quality scores (from 3 to 8) based on chemical features of red wine samples. Each model was trained, tuned, and evaluated to see which performed best.

## Dataset  

- **File**: `winequality-red-5.csv`  
- **Target**: Wine quality (3–8)  
- **Features**: Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, and alcohol  
- **Preprocessing**: Used `StandardScaler` to normalize features for KNN and ANN models  

## Models  

### 1. Artificial Neural Network (ANN)  
- **Model**: MLPClassifier  
- **Best setup**:  
  - Hidden layers: (125, 60)  
  - Alpha: 5e-05  
  - Learning rate: 0.02  
  - Activation: ReLU  
  - Solver: Adam  
- **Performance**: Accuracy 60.94%, Macro-F1 0.27  

### 2. Random Forest  
- **Best setup**:  
  - 100 estimators  
  - Max depth: 20  
  - min_samples_split: 5  
  - min_samples_leaf: 1  
- **Performance**: Accuracy 68.13%, Macro-F1 0.34  
- **Notes**: Alcohol, sulphates, and volatile acidity were the top predictors  

### 3. K-Nearest Neighbors (KNN)  
- **Best setup**:  
  - k = 12  
  - Weights: distance  
  - Metric: Euclidean  
- **Performance**: Accuracy 65.31%, Macro-F1 0.39  

## Comparison  

| Model | Accuracy | Macro-F1 |
|-------|-----------|-----------|
| Random Forest | 68.13% | 0.34 |
| KNN | 65.31% | 0.39 |
| ANN | 60.94% | 0.27 |

**Highlights:**  
- Random Forest had the highest accuracy.  
- KNN scored best on Macro-F1, handling class imbalance slightly better.  
- All models were biased toward the majority classes (5 and 6).  

## Structure  

ML-wine-classifier/
├── main.ipynb
├── ANN.ipynb
├── Random_Forest.ipynb
├── KNN.ipynb
├── winequality-red-5.csv
├── requirements.txt
└── README.md

markdown
Copy code

## Key Work  

- Exploratory analysis: heatmaps, distributions, and feature comparisons  
- Model tuning via grid and randomized search  
- Evaluation using accuracy, F1, and confusion matrices  
- Visualizations for loss, accuracy, learning curves, and feature importance  
- Analysis of how class imbalance affected results  

## Setup  

```bash
git clone <repository-url>
cd ML-wine-classifier
pip install -r requirements.txt
jupyter notebook
Dependencies
numpy, pandas, matplotlib, seaborn, scikit-learn, scipy

## Summary
The Random Forest model gave the best overall performance, while KNN handled class imbalance slightly better. The ANN was less accurate but showed stable learning behavior. All models struggled with minority classes due to the skewed dataset.