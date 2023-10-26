# Housing Price Prediction Model

This repository contains an exploration and implementation of various machine learning models to predict housing prices. The project was developed as part of the Machine Learning course at the Carlos III University of Madrid.

## Overview

The project was centered around understanding the relationship between various input variables and housing prices. The primary goal was to create a model that could predict the price of houses based on these input features.

### Data Preprocessing

- **One-Hot Encoding**: Applied to all categorical variables. Irrelevant variables were discarded.
- **Normalization**: Used Z-Score normalization on real (float) and integer data. However, special treatment was given to the price data due to its specific distribution.
- **Visualizations**: Functions to visualize distributions and relations between variables were utilized.
- **Correlation Matrix**: Established to visualize relationships among variables.
- **Data Splitting**: The dataset was divided into training (80%) and testing (20%) samples.

### Feature Engineering

A key component that set our model apart was the exploration of non-linear relations between features in our dataset. This exploration allowed us to create a unique set of input features, especially with a combination of bathrooms, bedrooms, and ratings (aseos+hab*rating) showing a linear correlation with the price.

### Model Exploration

- **Baseline Model**: Linear regression from scikit-learn was used as the baseline. Its performance was surprisingly competitive.
- **Other Models Tested**: RandomForestRegressor, ElasticNet, Lasso, Ridge, DecisionTreeRegressor, KNeighborsRegressor, GradientBoostRegressor, AdaBoostRegressor, and CatBoostRegressor.
- **Model Optimization**: Hyperparameters for GradientBoostRegressor and CatBoostRegressor were particularly optimized.

### Metrics

- **MAE (Mean Absolute Error)**: Used to measure the average of the absolute difference between the predicted and actual values.
- **MAPE (Mean Absolute Percentage Error)**: Offered a percentage error which is more interpretable and scale-independent.

### Results

The standout model in terms of performance was CatBoostRegresso, achieving MAPE values around 10-12% on the sample test set. In the competition held in class, its MAPE was of 12.91%, the best result among all groups, 0.64% better than the second best group. 

## Usage

You can test the model on your own by running the following command:

```bash
cd validator
python validator.py
```

## References

- [Artículo sobre balanceo de clases en aprendizaje automático](https://proceedings.mlr.press/v74/branco17a/branco17a.pdf)
- [Random Oversampling y Undersampling para clasificación con clases desequilibradas](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)
- [Handling Imbalanced Data by Oversampling with SMOTE and its Variants](https://medium.com/analytics-vidhya/handling-imbalanced-data-by-oversampling-with-smote-and-its-variants-23a4bf188eaf)
- [Notas sobre cómo manejar datos desequilibrados](https://reinec.medium.com/my-notes-handling-skewed-data-5984de303725)
- [Artículos sobre datos desequilibrados](https://medium.com/tag/skewed-data)
- [Libro: "An Introduction to Statistical Learning: with Applications in R" por James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013)](https://www.springer.com/gp/book/9781461471370)

