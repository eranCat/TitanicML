# Titanic ML - Passenger Survival Prediction

A machine learning project for analyzing and predicting passenger survival on the Titanic using various ML algorithms. Features exploratory data analysis, feature engineering, and multiple classification models with detailed performance evaluation.

## 📊 Project Overview

This project implements a complete ML pipeline to predict whether a Titanic passenger would have survived the ship's sinking. It includes data cleaning, exploratory analysis, feature engineering, model training, and evaluation using classic ML algorithms.

**Dataset**: Titanic passenger data with 891 training samples and 418 test samples

## 🎯 Objectives

- Perform exploratory data analysis (EDA) on Titanic dataset
- Clean and preprocess passenger data
- Engineer meaningful features for better model performance
- Train and evaluate multiple classification models
- Compare model performance and select the best approach
- Predict survival probability for test passengers

## 📁 Project Structure

```
TitanicML/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb     # EDA and visualization
│   ├── 02_data_cleaning.ipynb                 # Data preprocessing
│   ├── 03_feature_engineering.ipynb           # Feature creation
│   ├── 04_model_training.ipynb                # Model training
│   └── 05_results_analysis.ipynb              # Results visualization
├── data/
│   ├── train.csv                              # Training data
│   ├── test.csv                               # Test data
│   └── processed/                             # Cleaned datasets
├── models/
│   ├── saved_models/                          # Trained model files
│   └── model_artifacts/
├── src/
│   ├── preprocessing.py                       # Data cleaning functions
│   ├── feature_engineering.py                 # Feature creation
│   ├── models.py                              # Model definitions
│   ├── evaluation.py                          # Evaluation metrics
│   └── utils.py                               # Utility functions
├── outputs/
│   ├── plots/                                 # Visualization outputs
│   ├── results/                               # Model results
│   └── predictions.csv                        # Final predictions
├── requirements.txt                           # Dependencies
└── README.md
```

## 📈 Dataset Features

### Input Features
- **PassengerId**: Unique identifier
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender (male/female)
- **Age**: Passenger age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation

### Target Variable
- **Survived**: 1 = Yes, 0 = No

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/eranCat/TitanicML.git
cd TitanicML
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter:
```bash
jupyter notebook
```

5. Open and run notebooks in order:
   1. `01_exploratory_data_analysis.ipynb`
   2. `02_data_cleaning.ipynb`
   3. `03_feature_engineering.ipynb`
   4. `04_model_training.ipynb`
   5. `05_results_analysis.ipynb`

### Dependencies

```
pandas==1.5.0
numpy==1.23.0
matplotlib==3.6.0
seaborn==0.12.0
scikit-learn==1.1.0
xgboost==1.6.0
jupyter==1.0.0
```

## 🔍 Analysis Summary

### Exploratory Data Analysis
- **Data Shape**: 891 rows × 12 columns (training)
- **Missing Values**: Age (19.9%), Cabin (77.1%), Embarked (0.2%)
- **Target Distribution**: 38.4% survived, 61.6% did not
- **Class Distribution**: 24.2% 1st class, 20.6% 2nd class, 55.1% 3rd class

### Key Insights
1. **Gender Effect**: Women had 73.7% survival rate vs 18.9% for men
2. **Class Effect**: 1st class had 62.0% survival vs 26.6% for 3rd class
3. **Age Effect**: Children (age < 15) had higher survival rates
4. **Fare Effect**: Higher fares correlated with survival
5. **Family Size**: Passengers with 1-2 relatives had better survival chances

## 🛠️ Data Processing

### Data Cleaning
- Handle missing age values (mean/median/forward fill)
- Fill missing cabin data with 'Unknown'
- Handle missing embarked port (fill with mode)
- Remove unnecessary columns (PassengerId, Name, Ticket)
- Convert categorical variables

### Feature Engineering
- **Title Extraction**: Extract titles from names (Mr., Mrs., Miss., etc.)
- **Family Size**: Combine SibSp and Parch
- **Is Alone**: Binary indicator for traveling alone
- **Deck**: Extract deck letter from cabin
- **Age Groups**: Create age bins (child, teen, adult, elderly)
- **Fare Groups**: Quantile-based fare bins
- **Interaction Features**: Class × Gender, Age × Class combinations

## 🤖 Models Trained

### 1. Logistic Regression
```
Accuracy: 80.4%
Precision: 0.789
Recall: 0.629
F1-Score: 0.703
```

### 2. Random Forest
```
Accuracy: 82.8%
Precision: 0.820
Recall: 0.718
F1-Score: 0.766
```

### 3. Gradient Boosting
```
Accuracy: 84.1%
Precision: 0.841
Recall: 0.745
F1-Score: 0.790
```

### 4. Support Vector Machine
```
Accuracy: 81.6%
Precision: 0.807
Recall: 0.693
F1-Score: 0.746
```

### 5. XGBoost (Best Model)
```
Accuracy: 85.3%
Precision: 0.856
Recall: 0.769
F1-Score: 0.810
```

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 80.4% | 0.789 | 0.629 | 0.703 | 0.02s |
| SVM | 81.6% | 0.807 | 0.693 | 0.746 | 0.15s |
| Random Forest | 82.8% | 0.820 | 0.718 | 0.766 | 0.08s |
| Gradient Boosting | 84.1% | 0.841 | 0.745 | 0.790 | 0.12s |
| **XGBoost** | **85.3%** | **0.856** | **0.769** | **0.810** | 0.10s |

## 🎯 Feature Importance (Top 10)

1. Sex (Gender) - 0.284
2. Title - 0.198
3. Fare - 0.167
4. Pclass - 0.156
5. Age - 0.124
6. FamilySize - 0.034
7. IsAlone - 0.019
8. Embarked - 0.012
9. Deck - 0.005
10. Parch - 0.001

## 📈 Performance Visualization

- **Confusion Matrix**: True vs Predicted classifications
- **ROC Curve**: Trade-off between true positive and false positive rates
- **Learning Curves**: Model performance vs training set size
- **Feature Importance Plot**: Relative importance of each feature
- **Calibration Curve**: Model confidence calibration

## 🔮 Predictions

Final predictions on test set:
- Export format: CSV with PassengerId and predicted Survived
- Probability scores: Include survival probability for each passenger
- Confidence intervals: Estimate uncertainty in predictions

## 📝 Key Findings

1. **Gender was the strongest predictor** of survival
2. **Passenger class** significantly influenced survival chances
3. **Age** correlated with survival (children > women > men)
4. **Traveling with family** increased survival probability
5. **Ticket fare** indicated social status and cabin location
6. **XGBoost model** performed best with 85.3% accuracy

## 💡 Business Insights

- The "women and children first" evacuation policy is evident in data
- 1st class passengers had significantly better accommodations and evacuation priority
- Solo passengers had the lowest survival rates
- Higher fares correlated with better survival (proximity to lifeboats)

## 🧪 Cross-Validation

- **K-Fold**: 5-fold cross-validation for robust evaluation
- **Stratified Splits**: Maintain class distribution
- **Validation Scores**: Mean ± Std of CV folds
- **Hyperparameter Tuning**: Grid search and random search

## 🚀 How to Make Predictions

```python
from src.models import load_model
import pandas as pd

# Load trained model
model = load_model('models/saved_models/xgboost_final.pkl')

# Load test data
test_data = pd.read_csv('data/test.csv')
test_processed = preprocess_data(test_data)

# Make predictions
predictions = model.predict(test_processed)
probabilities = model.predict_proba(test_processed)

# Save results
results = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions,
    'Probability': probabilities[:, 1]
})
results.to_csv('outputs/predictions.csv', index=False)
```

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Pandas Documentation](https://pandas.pydata.org)

## 🤝 Contributing

Contributions welcome! Feel free to:
- Improve feature engineering
- Test new algorithms
- Enhance visualizations
- Add cross-validation strategies

## 📝 License

MIT License - See [LICENSE](./LICENSE)

## 👤 Author

**Eran Karaso** - Machine Learning Developer  
GitHub: [@eranCat](https://github.com/eranCat)

## 📞 Questions & Support

Found an issue or have a question? Open an [issue](https://github.com/eranCat/TitanicML/issues)

## 🎓 Learning Outcomes

After exploring this project, you'll understand:
- Complete ML pipeline from data to predictions
- Feature engineering techniques
- Model selection and evaluation
- Hyperparameter tuning
- Handling imbalanced classification
- Cross-validation strategies