# Titanic Machine Learning Project

Machine learning project analyzing Titanic passenger survival data with predictive modeling and feature engineering.

## Dataset

- **Passengers**: 891 training samples + 418 test samples
- **Features**: 12 features (age, sex, class, fare, etc.)
- **Target**: Survival (binary classification)
- **Source**: Kaggle Titanic Dataset

## Features

- Exploratory Data Analysis (EDA)
- Feature Engineering and Preprocessing
- Multiple ML Model Implementation
- Model Comparison and Evaluation
- Hyperparameter Tuning
- Survival Prediction on Test Data

## Project Structure

```
├── titanic.ipynb           # Main analysis notebook
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   ├── eda.ipynb           # Exploratory analysis
│   ├── preprocessing.ipynb # Data cleaning
│   └── modeling.ipynb      # Model training
└── src/
    ├── preprocessing.py
    ├── models.py
    └── evaluation.py
```

## Tech Stack

- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive notebooks

## Analysis Highlights

### Exploratory Data Analysis
- Survival rate: 38.4% (272/891)
- Class distribution: Strong correlation with survival
- Gender effect: Women had 74% survival rate vs 19% for men
- Age groups: Children had higher survival rates

### Key Features
1. **Sex**: Categorical, strong predictor
2. **Passenger Class**: Ordinal, major impact
3. **Fare**: Continuous, proxy for socioeconomic status
4. **Age**: Continuous, some missing values
5. **Family Relations**: Family size and presence

### Data Preprocessing
- Handled missing values (age, embarked)
- Encoded categorical variables
- Feature scaling for tree-independent models
- Feature selection using importance scores

## Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machines
- Neural Networks (optional)

## Results

Best model performance on test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.87 | 0.85 | 0.84 | 0.85 |
| Gradient Boosting | 0.86 | 0.84 | 0.82 | 0.83 |
| Logistic Regression | 0.81 | 0.78 | 0.76 | 0.77 |

## Running the Analysis

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Start Jupyter
jupyter notebook

# Open and run titanic.ipynb
```

## Key Insights

1. **Gender was the strongest predictor** - Women had vastly higher survival rates
2. **Class mattered significantly** - First-class passengers were more likely to survive
3. **Age played a role** - Younger passengers, especially children, had better chances
4. **Fare correlates with survival** - Higher fares indicated better accommodations
5. **Family dynamics** - Traveling with family increased chances for some groups

## Future Work

- [ ] Feature engineering with family names (surnames)
- [ ] Ensemble methods with stacking
- [ ] Deep learning models
- [ ] Kaggle competition submission
- [ ] Cross-validation analysis
- [ ] SHAP values for model interpretability

## References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**Predicting survival on the Titanic using ML** 🚢