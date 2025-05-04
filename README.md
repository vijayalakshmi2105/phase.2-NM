# Credit Card Fraud Detection Using Machine Learning

**Student Name**: Vijayalakshmi G  
**Register Number**: 412723205056  
**Institution**: Tagore Engineering College  
**Department**: B.Tech Information Technology  

---

## 1. Problem Statement

Credit card fraud is a serious and growing threat in the financial industry, leading to billions of dollars in losses each year. As online transactions increase, fraudsters use sophisticated tactics that are difficult to detect with traditional rule-based systems.

### Refined Problem
To develop a **machine learning-based solution** that detects fraudulent transactions in real-time, minimizing false positives while ensuring legitimate transactions are unaffected.

### Problem Type
Binary Classification — Classify each transaction as either **fraudulent** or **legitimate**.

### Why It Matters
Real-time fraud detection:
- Protects users and institutions from financial losses  
- Strengthens customer trust  
- Improves efficiency in digital banking and e-commerce  

---

## 2.  Project Objectives

### Core Goals
- Detect fraudulent credit card transactions using machine learning  
- Reduce false positive and false negative rates  
- Prioritize interpretability for feature insights

### Technical Objectives
- Train and compare **at least two models** (e.g., Random Forest, Logistic Regression)  
- Optimize **accuracy**, **precision**, and **recall** while avoiding overfitting  
- Address class imbalance using techniques like **SMOTE** and **class weighting**

### Evolved Goals
Following EDA, the focus shifted toward **handling class imbalance** and **improving recall**, due to the rarity of fraud cases.

---

## 3. Project Workflow

  -- Data Collection → Data Preprocessing → EDA → Feature Engineering → Model Building → Model Evaluation
  
---

## 4. Data Description

- **Dataset Name**: Credit Card Fraud Detection  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Data Type**: Structured tabular data  
- **Records**: 284,807 transactions  
- **Features**: 30 (28 anonymized + `Time`, `Amount`)  
- **Target Variable**: `Class` (0 = legitimate, 1 = fraud)  
- **Nature**: Static Dataset  

---

## 5. Data Preprocessing

- **Missing Values**: None  
- **Duplicates**: Not found  
- **Outliers**: Detected and handled using log transformation and IQR  
- **Type Conversion**: Verified and corrected as needed  
- **Encoding**: Not required (all numerical)  
- **Scaling**: Standardized `Amount` and `Time` using `StandardScaler`

---

## 6. Exploratory Data Analysis (EDA)

### Univariate Analysis
- Fraud class represents **~0.17%** of the dataset  
- `Amount` and `Time` are **skewed**

### Bivariate & Multivariate Analysis
- Fraudulent transactions tend to have distinct amount patterns  
- Correlation heatmap revealed strong feature interrelations

### Key Insights
- The dataset is **highly imbalanced**  
- Some anonymized features show strong **class separation**

---

## 7. Feature Engineering

- **New Features**: None added due to anonymized data  
- **Transformations**:  
  - Scaled `Amount` and `Time`  
  - Created a binary indicator for high-value transactions *(optional)*  
- **Dimensionality Reduction**: Not required (PCA already applied)

---

## 8.Model Building

### Models Used
- **Logistic Regression**: Simple and interpretable  
- **Random Forest**: Captures non-linear patterns and handles overfitting

### Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique)  
- **Class Weight Adjustment**

### Data Split
- 80:20 stratified train-test split

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

---

## 9. Model Results & Visualization

- **Confusion Matrix**: Visual performance breakdown  
- **ROC Curve**: Threshold performance evaluation  
- **Precision-Recall Curve**: Better suited for imbalanced classes  
- **Feature Importance (Random Forest)**: Highlights most predictive features

### Interpretation
- Helps identify top influencing features  
- Improves **trust** and **transparency** of predictions  

---

## 10. Tools and Technologies

- **Language**: Python  
- **Notebook**: Jupyter, Google Colab  
- **Libraries**:  
  - `pandas`, `numpy`  
  - `scikit-learn`, `imbalanced-learn`  
  - `matplotlib`, `seaborn`
      
- **Optional Dashboard**: Streamlit (for real-time fraud detection interface)

---

## 11. Team Members and Contributions

| Name             | Contribution                        |
|------------------|-------------------------------------|
| Mohanapriya.k    | Data Cleaning, Preprocessing        |
| Varsha.T         | EDA, Visualization                  |
| Rajitha.M        | Model Development & Evaluation      |
| Vijayalakshmi.g  | Report Writing, GitHub Management   |


