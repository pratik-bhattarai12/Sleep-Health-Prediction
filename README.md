# Sleep Health Prediction

A machine learning project focused on predicting sleep health metrics and patterns using data-driven analysis.

## Overview

This repository contains Jupyter notebooks and analysis for building predictive models related to sleep health. The project explores sleep patterns, health metrics, and develops machine learning models to forecast sleep quality and related health outcomes. Using a dataset of 100,000 records with 32 features, this project implements comprehensive exploratory data analysis and multiple regression models to predict sleep quality scores.

## Dataset

- **Size**: 100,000 records
- **Features**: 32 attributes including demographic, lifestyle, and health metrics
- **Target Variable**: `sleep_quality_score`
- **Key Attributes**: Age, gender, occupation, BMI, sleep duration, REM percentage, deep sleep percentage, stress score, cognitive performance, and more

## Exploratory Data Analysis (EDA) & Visualization

The EDA phase involved comprehensive data exploration and visualization techniques:

### Data Inspection
- Loaded and examined dataset structure (shape: 100,000 × 32)
- Reviewed column names and data types
- Inspected first, last, and random records to understand data distribution
- Checked for missing values and data quality issues

### Visualization Techniques Used
- **Histograms & Distributions** - Analyzed distribution of sleep quality scores, sleep duration, and other continuous variables
- **Scatter Plots** - Explored relationships between sleep quality and key predictors (sleep duration, stress score, exercise)
- **Correlation Heatmaps** - Identified feature relationships and correlations using seaborn heatmaps
- **Box Plots & Violin Plots** - Analyzed sleep quality across categorical variables (gender, occupation, season, day type)
- **Statistical Summaries** - Calculated mean, median, std, min, and max for all features

### Key EDA Findings
- Distribution analysis of sleep quality scores and sleep characteristics
- Correlation patterns between lifestyle factors and sleep quality
- Impact of demographic factors (age, gender, BMI) on sleep health
- Seasonal and day-type variations in sleep patterns
- Relationship between stress levels and cognitive performance with sleep quality

## Machine Learning Models

The project implements and compares three regression models to predict sleep quality scores:

### 1. Linear Regression
- **Purpose**: Baseline model for linear relationship prediction
- **Approach**: Used scikit-learn's `LinearRegression` 
- **Use Case**: Provides simple, interpretable baseline for comparison
- **Advantages**: Fast, easy to interpret, good for understanding feature importance
- **Output**: Continuous predictions of sleep quality scores

### 2. Random Forest Regressor
- **Purpose**: Capture non-linear relationships and feature interactions
- **Approach**: Ensemble method combining multiple decision trees
- **Implementation**: Used scikit-learn's `RandomForestRegressor`
- **Advantages**: 
  - Handles non-linear patterns well
  - Provides feature importance rankings
  - Robust to outliers
  - Generally produces better accuracy than linear models
- **Output**: More accurate continuous predictions

### 3. Decision Tree Regressor
- **Purpose**: Capture hierarchical decision patterns
- **Approach**: Single tree model with if-then rules
- **Implementation**: Used scikit-learn's `DecisionTreeRegressor`
- **Advantages**: 
  - Highly interpretable
  - Captures feature interactions
  - No feature scaling required
  - Good for understanding key splitting features
- **Output**: Decision rules for sleep quality prediction

### Model Development Workflow
1. **Train-Test Split**: Used `train_test_split` to divide data (typically 80-20 split)
2. **Feature Encoding**: Applied `LabelEncoder` and `OneHotEncoder` for categorical variables
3. **Model Training**: Fitted each model on training data
4. **Predictions**: Generated predictions on test set
5. **Evaluation Metrics**:
   - **R² Score** - Measure of model fit quality
   - **Mean Squared Error (MSE)** - Average squared prediction errors
   - **Mean Absolute Error (MAE)** - Average absolute prediction errors
   - **Accuracy Score** - Overall prediction accuracy

### Model Performance Comparison
- Linear Regression: Fast but limited to linear relationships
- Random Forest: Best overall performance with accurate predictions
- Decision Tree: Good interpretability with reasonable performance

## Project Structure

```
Sleep-Health-Prediction/
├── sleep_health_prediction2.ipynb    # Main analysis notebook
├── sleep_health_dataset.csv          # Dataset (100K records)
├── sleep_project_flowchart.png       # Project workflow diagram
├── CET140 AS1_Research_Report.pdf    # Detailed research report
└── README.md                         # This file
```

## Key Features

- **Comprehensive exploratory data analysis** of sleep health data with multiple visualization techniques
- **Feature engineering and data preprocessing** including encoding categorical variables
- **Multiple regression models** (Linear, Random Forest, Decision Tree) with comparison
- **Performance evaluation** using multiple metrics (R², MSE, MAE, Accuracy)
- **Visualization of sleep patterns** and correlations with health factors
- **Interpretability** of model predictions and feature importance

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook

### Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (r2_score, mean_squared_error, 
                             mean_absolute_error, accuracy_score)
```

### Installation & Execution
1. Clone the repository
2. Install required dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open `sleep_health_prediction2.ipynb` in Jupyter Notebook
4. Run all cells to execute the complete analysis pipeline

### Usage
- Run the notebook cell-by-cell to follow the analysis workflow
- Modify hyperparameters in model sections to experiment
- Adjust visualization parameters to create custom plots
- Test with different feature combinations

## Analysis Workflow

1. **Data Loading** - Load CSV file into pandas DataFrame
2. **Data Exploration** - Inspect structure, columns, and sample records
3. **Statistical Analysis** - Calculate descriptive statistics
4. **Visualization** - Create plots and heatmaps to understand patterns
5. **Data Preprocessing** - Encode categorical variables, handle missing values
6. **Feature Selection** - Identify important features
7. **Model Training** - Train Linear, Random Forest, and Decision Tree models
8. **Model Evaluation** - Calculate R², MSE, MAE, and accuracy metrics
9. **Comparison** - Compare model performance and interpret results

## Technology Stack

- **Python** - Primary programming language
- **Jupyter Notebook** - Interactive analysis and visualization
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Basic data visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms and evaluation

## Results & Insights

- Sleep quality is influenced by multiple factors including sleep duration, stress levels, and lifestyle habits
- Random Forest model provides superior predictive performance compared to linear baselines
- Seasonal variations and day-type (weekday vs weekend) impact sleep patterns
- Feature importance analysis reveals key factors affecting sleep quality
- Demographic factors (age, BMI) show notable correlations with sleep health

## Contributing

Feel free to contribute improvements, additional analyses, or enhancements to the project. Possible areas for improvement:
- Cross-validation for robust model evaluation
- Hyperparameter tuning for better performance
- Additional ML models (XGBoost, Gradient Boosting)
- Time-series analysis if temporal data is available
- Feature engineering for new derived features

## License

[Add your license information here]

## References

- Research Report: CET140 AS1_Research_Report.pdf
- Project Flowchart: sleep_project_flowchart.png
- Dataset: sleep_health_dataset.csv
