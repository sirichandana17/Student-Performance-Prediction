# 🎓 Student Performance Prediction

A machine learning–based student performance prediction system that classifies students as **Pass or Fail** using academic, demographic, and behavioral features, with detailed evaluation and visualization.

---

## 📌 Project Overview

This project predicts whether a student will **pass or fail** based on data from the **Student Performance Dataset** using a **Support Vector Machine (LinearSVC)** model.  
It evaluates performance under **three real-world scenarios** by selectively including or excluding previous exam scores.

---

## 🚀 Key Features

- Binary **Pass/Fail prediction**
- Three predictive scenarios:
  1. Model knowing **G1 & G2 scores**
  2. Model knowing **only G1 score**
  3. Model **without any previous scores**
- **Chi-Square feature selection**
- **Cross-validation** for reliable accuracy
- **Confusion matrix analysis**
- Visualizations:
  - Confusion Matrix
  - Pass/Fail prediction distribution
- Performance metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - False Pass Rate
  - False Fail Rate

---

## 🧠 Machine Learning Pipeline

- Label encoding for categorical features
- Binary conversion of grades (Pass ≥ 10)
- Feature selection using **SelectKBest (Chi-Square)**
- Classification using **Linear Support Vector Machine**
- Evaluation with 5-Fold Cross Validation

---

## 📊 Scenarios Evaluated

| Scenario | Description |
|--------|------------|
| Scenario 1 | Model knows G1 & G2 scores |
| Scenario 2 | Model knows only G1 score |
| Scenario 3 | Model has no previous score information |

---

## 📂 Dataset

- File: `student-mat.csv`
- Source: UCI Machine Learning Repository
- Separator: `;`
- Target Variable: `G3` (Final Grade → Pass/Fail)

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
---

## ▶️ How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
python SPP-ML.py
```
---

## 📈 Output
- Cross-validation accuracy
- Test accuracy
- Classification report
- Confusion matrix visualization
- Pass/Fail prediction bar chart
  
---

## 🎯 Applications
- Student performance monitoring
- Early identification of at-risk students
- Educational data analysis
- Machine learning academic project

---

## 👩‍💻 Author
- Siri Chandana Kanaparthi
- B.Tech CSE (AI & ML)
---
