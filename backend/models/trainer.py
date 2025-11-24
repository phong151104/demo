"""
Model Training Module
Handles training of various machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def train_model(X_train, y_train, X_test, y_test, model_type, params=None):
    """
    Train a machine learning model based on the specified type and parameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_type : str
        Type of model to train
    params : dict
        Model parameters
        
    Returns:
    --------
    model : object
        Trained model object
    metrics : dict
        Dictionary of evaluation metrics
    """
    if params is None:
        params = {}

    model = None
    
    try:
        if model_type == "Logistic Regression":
            model = LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 200),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        elif model_type == "LightGBM":
            model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', -1),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
            
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(
                iterations=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=0,
                allow_writing_files=False
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return model, metrics
        
    except Exception as e:
        raise Exception(f"Error training {model_type}: {str(e)}")
