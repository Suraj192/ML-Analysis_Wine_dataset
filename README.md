# Wine Quality Prediction – Machine Learning Pipeline

## 📌 Project Overview

This project implements an end-to-end machine learning pipeline to predict wine quality using physicochemical properties.

The objective is to demonstrate structured ML development including preprocessing, model training, and evaluation using Random Forest.

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier

---

## 📂 Project Structure

```
ml_analysis_wine_Dataset/
│
├── data/
│   └── wine.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline Steps

1. Data loading from CSV
2. Data preprocessing
3. Train-test split
4. Random Forest model training
5. Model evaluation using accuracy and classification report

---

## 🚀 How to Run

1. Clone the repository:
```
git clone <your-repo-url>
cd ML-Analysis_Wine_dataset
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the pipeline:
```
python main.py
```

---

## 📊 Model

Random Forest Classifier  
- n_estimators = 100  
- random_state = 42  

---

## 🎯 Purpose

This project demonstrates modular ML project structuring following production-style code organization rather than notebook-only experimentation.
