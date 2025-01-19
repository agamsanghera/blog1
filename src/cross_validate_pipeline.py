from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import pandas as pd

def cross_validate_pipeline(X_train, y_train, cv, return_train_score, *steps):
    """
    Performs cross-validation on a given training set, given the steps in a pipeline.
    
    Parameters:
    - X_train: Training set.
    - y_train: Labels for the training set.
    - cv: Number of cross-validation folds.
    - return_train_score: Whether to return the training scores.
    - steps: Variable number of steps used to create a pipeline.
    
    Returns:
    - cv_results: A DataFrame containing the cross-validation results.
    """
    
    if not steps:
        raise ValueError("At least one step is required to build a pipeline.")
    if not isinstance(cv, int):
        raise ValueError("cv value needs to be an integer.")
    
    model_pipeline = Pipeline(steps=steps)
    
    cv_results = cross_validate(
        model_pipeline, 
        X_train, 
        y_train, 
        cv=cv, 
        return_train_score=return_train_score
    )
    
    return pd.DataFrame(cv_results)
