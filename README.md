<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F766E,50:14B8A6,100:2563EB&height=220&section=header&text=Behavioral%20Mental%20Health%20Risk%20Assessment&fontSize=34&fontColor=ffffff&fontAlignY=50&animation=fadeIn"/>

</div>

---

# Behavioral Mental Health Risk Assessment using Machine Learning and Social Media Analytics

This project presents an end-to-end machine learning framework for behavioral mental health risk assessment using adolescent social media and lifestyle indicators. The pipeline integrates exploratory data analysis, feature engineering, data preprocessing, imbalance learning, comparative model development, hyperparameter optimization, explainable machine learning, and comprehensive performance evaluation to support reliable, interpretable, and data-driven mental health risk identification.

<div align="left">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Visualization-4C72B0?style=flat)](https://seaborn.pydata.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-EC6B23?style=flat)](https://xgboost.readthedocs.io/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-0A7B83?style=flat)](https://imbalanced-learn.org/)
[![Machine Learning](https://img.shields.io/badge/AI-Machine_Learning-2563EB?style=flat)](#)
[![Healthcare AI](https://img.shields.io/badge/Domain-Mental_Health_AI-059669?style=flat)](#)
[![Explainable AI](https://img.shields.io/badge/XAI-Feature_Interpretability-7C3AED?style=flat)](#)
[![License](https://img.shields.io/badge/License-MIT-4B5563?style=flat)](https://opensource.org/licenses/MIT)

</div>

---

# Abstract

Mental health disorders among adolescents have become an increasingly important public health challenge, with behavioral patterns observed through social media and lifestyle characteristics providing valuable indicators for early risk assessment. This project introduces a comprehensive machine learning pipeline for behavioral mental health risk assessment that combines exploratory data analysis, advanced preprocessing techniques, feature engineering, imbalance-aware learning, predictive modeling, and explainable artificial intelligence.

Instead of relying on a single classification algorithm, the framework systematically compares multiple machine learning models, evaluates their predictive performance using diverse classification metrics, and analyzes feature importance to improve both predictive accuracy and model transparency. The resulting workflow demonstrates how modern machine learning techniques can support interpretable and data-driven decision making for behavioral mental health assessment.

---


# Table of Contents

1. [Overview](#-overview)

2. [Machine Learning Pipeline](#-machine-learning-pipeline)

   - Dataset Preparation
   - Feature Engineering
   - Imbalanced Learning
   - Model Development
   - Hyperparameter Optimization
   - Model Selection

3. [System Architecture](#system-architecture)

4. [Experimental Workflow](#experimental-workflow)

5. [Model Evaluation](#-model-evaluation)

   - Comparative Performance
   - Explainability
   - Experimental Results

6. [Project Structure](#-project-structure)

7. [Installation](#-installation)

8. [License](#license)

9. [Author](#author)

10. [Support](#-support)

---

# 📌 Overview

Mental health disorders among adolescents have become an increasingly important public health concern, motivating the development of intelligent data-driven approaches capable of identifying behavioral risk patterns at early stages. Machine learning provides an effective framework for discovering complex relationships between behavioral characteristics and mental health outcomes while supporting objective and reproducible decision making.

This project implements a complete behavioral mental health analytics pipeline rather than a standalone classification model. Beginning with raw behavioral observations, the workflow performs exploratory data analysis, data preprocessing, feature engineering, imbalance-aware learning, comparative model development, and explainable evaluation before selecting the most reliable predictive model.

The framework emphasizes three complementary objectives:

- **Predictive Performance** through comparative machine learning and hyperparameter optimization.

- **Model Robustness** through comprehensive preprocessing and imbalance-aware learning.

- **Model Interpretability** through explainable machine learning and feature importance analysis.

Unlike traditional predictive workflows that focus primarily on classification accuracy, this project adopts a comprehensive experimental methodology that prioritizes reproducibility, transparency, and systematic evaluation across multiple machine learning algorithms.

---

# System Architecture

The proposed framework follows a layered machine learning architecture that transforms raw behavioral observations into interpretable mental health risk predictions through a sequence of preprocessing, learning, evaluation, and explainability stages. Each layer is designed to ensure reproducibility, modularity, and transparency while enabling objective comparison among multiple predictive models.

```mermaid
flowchart TB

subgraph Data Layer

A[Behavioral Mental Health Dataset]

B[Social Media Behavior]

C[Sleep & Lifestyle]

D[Academic Performance]

E[Daily Activities]

F[Psychological Indicators]

end

subgraph Data Processing Layer

G[Exploratory Data Analysis]

H[Missing Value Inspection]

I[Data Cleaning]

J[Categorical Encoding]

K[Feature Scaling]

L[Correlation Analysis]

M[Train-Test Split]

end

subgraph Learning Layer

N[Class Imbalance Analysis]

O[SMOTE Oversampling]

P[Balanced Training Data]

end

subgraph Machine Learning Layer

Q[Logistic Regression]

R[K-Nearest Neighbors]

S[Decision Tree]

T[Support Vector Machine]

U[Random Forest]

V[XGBoost]

end

subgraph Evaluation Layer

W[Classification Metrics]

X[ROC-AUC Analysis]

Y[Confusion Matrix]

Z[Cross Validation]

AA[Feature Importance]

end

subgraph Decision Layer

AB[Best Model Selection]

AC[Behavioral Mental Health Risk Assessment]

end

A --> G

B --> G

C --> G

D --> G

E --> G

F --> G

G --> H

H --> I

I --> J

J --> K

K --> L

L --> M

M --> N

N --> O

O --> P

P --> Q

P --> R

P --> S

P --> T

P --> U

P --> V

Q --> W

R --> W

S --> W

T --> W

U --> W

V --> W

W --> X

X --> Y

Y --> Z

Z --> AA

AA --> AB

AB --> AC
```

The proposed architecture follows a modular machine learning workflow that separates data preparation, model development, and evaluation into independent layers. This layered design improves reproducibility, facilitates objective comparison across multiple learning algorithms, and enhances model transparency through explainable feature analysis.

---

# Experimental Workflow

The complete experimental pipeline follows a reproducible sequence of preprocessing, predictive modeling, and model interpretation, enabling systematic comparison between multiple machine learning algorithms under identical experimental conditions.

```mermaid
flowchart LR

A[Raw Dataset]

A --> B[Exploratory Data Analysis]

B --> C[Data Cleaning]

C --> D[Missing Value Handling]

D --> E[Categorical Encoding]

E --> F[Feature Engineering]

F --> G[Feature Scaling]

G --> H[Correlation Analysis]

H --> I[Train / Test Split]

I --> J[Class Imbalance Handling]

J --> K[Model Training]

K --> L[Hyperparameter Optimization]

L --> M[Model Evaluation]

M --> N[Feature Importance Analysis]

N --> O[Best Model Selection]

O --> P[Behavioral Mental Health Risk Assessment]
```

---

# Machine Learning Pipeline

The project implements a complete predictive analytics workflow that transforms raw behavioral observations into interpretable mental health risk predictions. Every stage contributes to improving data quality, model robustness, predictive performance, and explainability.

### Dataset Preparation

The raw behavioral dataset undergoes extensive preprocessing before model development. Data quality assessment, categorical encoding, feature scaling, and train-test partitioning ensure that every classifier is trained under identical and reproducible experimental settings.

### Feature Engineering

Feature engineering improves the predictive representation of behavioral information by transforming raw variables into more informative learning features. Correlation analysis and statistical feature inspection are used to identify redundant attributes and strengthen model performance.

### Imbalanced Learning

Behavioral healthcare datasets frequently exhibit skewed class distributions. To mitigate prediction bias toward majority classes, the training data is balanced using SMOTE before model development, resulting in improved minority class recognition and more reliable evaluation.

### Model Development

Instead of relying on a single predictive model, the framework performs a comparative evaluation across multiple supervised learning algorithms.

The evaluated models include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- Extreme Gradient Boosting (XGBoost)

Hyperparameter optimization is subsequently applied to improve predictive performance while reducing overfitting and increasing model generalization.

---

# 📊 Model Evaluation

The proposed framework evaluates every classifier using multiple complementary performance metrics rather than relying solely on prediction accuracy. This provides a more comprehensive understanding of predictive behavior under imbalanced class distributions.

Evaluation focuses on:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-Validation
- Confusion Matrix Analysis

Beyond quantitative evaluation, the framework also investigates feature importance to better understand how behavioral indicators contribute to model predictions and to improve overall model transparency.

---

# 📈 Experimental Results

The notebook contains several visual analyses illustrating different stages of the experimental pipeline. Instead of presenting isolated performance numbers, the following visualizations summarize the most important findings obtained throughout the study.

## Comparative Model Performance

<p align="center">

<!-- Replace with the comparison figure generated in the notebook -->

<img src="figures/model_comparison.png" width="760"/>

</p>

Performance comparison across all evaluated classifiers demonstrates the impact of preprocessing, imbalance-aware learning, and hyperparameter optimization on predictive accuracy and overall model robustness.

---

## Confusion Matrix

<p align="center">

<!-- Replace with the best-performing model confusion matrix -->

<img src="figures/confusion_matrix.png" width="560"/>

</p>

The confusion matrix provides a detailed view of prediction outcomes and enables analysis of correctly and incorrectly classified samples beyond aggregate evaluation metrics.

---

## Feature Importance

<p align="center">

<!-- Replace with the notebook feature importance figure -->

<img src="figures/feature_importance.png" width="760"/>

</p>

Feature importance analysis improves interpretability by identifying the behavioral variables that contribute most significantly to mental health risk prediction, providing valuable insight into the decision-making process of the selected model.

---


---

# 📁 Project Structure

The repository is organized to ensure clarity, reproducibility, and ease of navigation across different components of the machine learning pipeline.

```text
Behavioral-Mental-Health-Risk-Assessment
│
├── behavioral_mental_health.ipynb
│
├── dataset/
│   └── teen_behavior_dataset.csv
│
├── figures/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── correlation_heatmap.png
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Installation

The following steps outline how to set up and run the project locally.

## 1. Clone Repository

```bash
git clone https://github.com/farzadjannati/Behavioral-Mental-Health-Risk-Assessment.git

cd Behavioral-Mental-Health-Risk-Assessment
```

---

## 2. Create Virtual Environment

```bash
conda create -n mental-health python=3.10

conda activate mental-health
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
behavioral_mental_health.ipynb
```

Execute all cells sequentially to reproduce the full machine learning pipeline.

---

# 📊 Final Insights

The experimental results demonstrate that behavioral signals extracted from adolescent social media and lifestyle data contain meaningful predictive patterns for mental health risk assessment.

Key observations include:

- Machine learning models benefit significantly from proper preprocessing and feature engineering.
- Class imbalance handling improves sensitivity toward minority classes.
- Ensemble models such as Random Forest and XGBoost consistently outperform baseline classifiers.
- Explainability techniques enhance interpretability and trust in predictive outcomes.
- Comparative evaluation is essential for selecting robust and generalizable models.

---

# 👨‍💻 Author

**Farzad Jannati**

M.Sc. Student – University of Tehran  
Research Assistant – Social Networks Lab  

### Research Interests:
- Machine Learning
- Data Mining
- Behavioral Analytics
- Explainable AI (XAI)
- Large Language Models (LLMs)

📧 Email: farzadjannati@ut.ac.ir  
💻 GitHub: https://github.com/farzadjannati  
💼 LinkedIn: https://linkedin.com/in/farzadjannati  

---

# ⭐ Support

If you find this project useful, consider giving it a star ⭐ on GitHub.

Your support helps improve visibility of open-source research projects and encourages further development in machine learning for healthcare applications.

---

# 📌 Footer

<p align="center">

Built with ❤️ using Python, Pandas, Scikit-Learn

</p>

---
