# tests/test_cross_validate_pipeline.py
# References: Tiffany's Breast Cancer Predictor and ChatGPT were used as references

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from src.cross_validate_pipeline import cross_validate_pipeline

def test_cross_validate_pipeline_normal():
    """
    Tests the function takes in the valid parameters and outputs a valid DataFrame.
    """
    X, y = make_classification(n_samples=100, n_features=5, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    result = cross_validate_pipeline(
        X_train,
        y_train, 
        5, 
        True, 
        ('preprocessor', StandardScaler()), 
        ('model', LogisticRegression(random_state=123, max_iter=1000))
    )
    
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame as return type"
    assert 'train_score' in result.columns, "Expected 'train_score' in DataFrame columns"
    assert 'test_score' in result.columns, "Expected 'test_score' in DataFrame columns"

def test_cross_validate_pipeline_no_steps():
    """
    Tests the function takes in an invalid parameter `step`.
    """
    X, y = make_classification(n_samples=100, n_features=5, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    with pytest.raises(ValueError, match="At least one step is required to build a pipeline."):
        cross_validate_pipeline(X_train, y_train, 5, True)

def test_cross_validate_pipeline_invalid_cv():
    """
    Tests the function takes in an invalid parameter `cv`.
    """
    X, y = make_classification(n_samples=100, n_features=5, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    with pytest.raises(ValueError, match="cv value needs to be an integer."):
        cross_validate_pipeline(X_train, y_train, [1,2,3], True)
