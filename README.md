# 🧠 Breast Cancer Metastasis Prediction using H2O AutoML

## 🚀 Overview

This project is developed as part of the **Women in Data Science (WiDS)** challenge. The goal is to build a robust **machine learning pipeline** to predict the **metastatic diagnosis period** for breast cancer patients using real-world healthcare and demographic data.

The project covers **end-to-end ML workflow** including data preprocessing, feature engineering, exploratory data analysis (EDA), and automated model building using **H2O AutoML**.

---

## 🎯 Problem Statement

Predict the **metastatic diagnosis period** (time until cancer metastasis) based on:

* Patient demographics
* Geographic and socio-economic features
* Medical diagnosis codes
* Environmental factors (temperature, population stats)

---

## 🧠 Tech Stack

* **Language:** Python
* **Libraries:**

  * NumPy, Pandas
  * Matplotlib, Seaborn
  * Scikit-learn
  * H2O AutoML
* **Tools:** Jupyter Notebook

---

## 📂 Dataset

The project uses:

* `train.csv` → Training dataset
* `test.csv` → Test dataset
* `solution_template.csv` → Submission format

---

## 🔄 Project Workflow

### 1. 📌 Data Preprocessing

* Combined train & test datasets for uniform processing
* Fixed inconsistencies in:

  * State and ZIP mappings
  * Diagnosis codes
* Handled missing values using:

  * Mean imputation (population features)
  * Category replacement (e.g., "Uninsured")
* Created missing value indicators:

  * `bmi_missing`, `patient_race_missing`
* Temperature data transformation using pivot & melt operations

---

### 2. 🧹 Feature Engineering

* BMI categorization into groups
* Outlier treatment:

  * BMI capped at ≤ 45
  * Rent burden cleaned
* Created new derived features:

  * Socio-economic bins (poverty, education)
* Dropped irrelevant columns:

  * IDs, constant columns, redundant features

---

### 3. 📊 Exploratory Data Analysis (EDA)

* Distribution of payer types (pie chart)
* Top diagnosis codes (bar plots)
* Relationship analysis:

  * Age vs metastatic period
  * Poverty vs diagnosis delay
* Feature trend visualization using Seaborn

---

### 4. 🤖 Model Building (Core Highlight)

Used **H2O AutoML** for automated model selection:

* Trained multiple models (max_models = 15)
* Additional run with time-based optimization
* Compared models using leaderboard

```python
aml = H2OAutoML(max_models=15, seed=42)
aml.train(y='metastatic_diagnosis_period', training_frame=train_data)
```

---

### 5. 📉 Prediction & Ensemble

* Generated predictions using:

  * Best model from first AutoML run
  * Best model from second AutoML run
* Combined predictions for final output

---

## 📊 Output

* Final predictions stored in:

  * `solution_template.csv`
* Ready for submission

---

## ⚡ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/dhanushgoudra2003/WiDS-project.git
cd WiDS-project
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn h2o
```

### 3. Run the notebook

```bash
jupyter notebook Code.ipynb
```

---

## 🌟 Key Highlights

✔ End-to-end ML pipeline
✔ Advanced data cleaning & preprocessing
✔ Feature engineering on real-world healthcare data
✔ Automated model selection using **H2O AutoML**
✔ Ensemble prediction strategy

---

## 🔥 Future Improvements

* Add evaluation metrics (RMSE, MAE)
* Hyperparameter tuning for top models
* Deploy using Streamlit
* Add feature importance analysis
* Improve ensemble strategy

---


## ⭐ Acknowledgment

This project is inspired by the **WiDS (Women in Data Science)** initiative and focuses on solving real-world healthcare prediction problems using machine learning.

---
