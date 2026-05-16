# 🚢 TitanicML

Machine learning project predicting Titanic passenger survival using classification algorithms and exploratory data analysis.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| Notebook | Jupyter |
| ML | scikit-learn |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |

## Features

- **Exploratory Data Analysis (EDA)** — visualizing survival rates by class, gender, age, and embarkation port
- **Feature engineering** — handling missing values, encoding categoricals, creating new features
- **Multiple models** — Logistic Regression, Random Forest, SVM, XGBoost
- **Model evaluation** — accuracy, precision, recall, confusion matrix, ROC curve
- **Kaggle-ready** — submission CSV generation

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~80% |
| Random Forest | ~83% |
| XGBoost | ~84% |

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Setup

```bash
git clone https://github.com/eranCat/TitanicML.git
cd TitanicML
pip install -r requirements.txt
jupyter notebook
```

Open `titanic.ipynb` and run all cells.

## Dataset

Uses the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data) (`train.csv`, `test.csv`).

## Author

**Eran Karaso** — [Portfolio](https://erancat.github.io/portfolio-site) · [GitHub](https://github.com/eranCat)
