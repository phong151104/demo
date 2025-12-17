"""
Data Balancing Module
Handles imbalanced datasets using various resampling techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def balance_data(
    data: pd.DataFrame,
    target_column: str,
    method: str = "SMOTE",
    random_state: int = 42,
    sampling_strategy: str = "auto",
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Balance imbalanced dataset using various resampling techniques
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target column
    method : str
        Balancing method: "SMOTE", "Random Over-sampling", "Random Under-sampling", "No Balancing"
    random_state : int
        Random state for reproducibility
    sampling_strategy : str or float
        Sampling strategy for resampling
    
    Returns:
    --------
    balanced_data : pd.DataFrame
        Balanced dataset
    info : Dict
        Information about the balancing process
    """
    
    try:
        # Validate inputs
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Get original class distribution
        original_dist = y.value_counts().to_dict()
        
        info = {
            'method': method,
            'original_distribution': original_dist,
            'original_size': len(data)
        }
        
        # No balancing
        if method == "No Balancing":
            info['balanced_distribution'] = original_dist
            info['balanced_size'] = len(data)
            info['message'] = "No balancing applied"
            return data, info
        
        # Random Over-sampling
        elif method == "Random Over-sampling":
            from imblearn.over_sampling import RandomOverSampler
            
            ros = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
        # Random Under-sampling
        elif method == "Random Under-sampling":
            from imblearn.under_sampling import RandomUnderSampler
            
            rus = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
        # SMOTE
        elif method == "SMOTE":
            from imblearn.over_sampling import SMOTE
            
            # Check if we have enough samples for SMOTE
            min_samples = y.value_counts().min()
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            if min_samples < 2:
                info['message'] = "Not enough samples for SMOTE. Using Random Over-sampling instead."
                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
                X_resampled, y_resampled = ros.fit_resample(X, y)
            else:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=k_neighbors
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Combine back into DataFrame
        balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
        
        # Get balanced class distribution
        balanced_dist = y_resampled.value_counts().to_dict()
        
        info['balanced_distribution'] = balanced_dist
        info['balanced_size'] = len(balanced_data)
        info['size_change'] = len(balanced_data) - len(data)
        info['message'] = f"Successfully balanced data using {method}"
        
        return balanced_data, info
        
    except ImportError as e:
        raise ImportError(
            f"imbalanced-learn library is required for data balancing. "
            f"Install it with: pip install imbalanced-learn. Error: {str(e)}"
        )
    except Exception as e:
        raise Exception(f"Error during data balancing: {str(e)}")


def get_class_distribution(data: pd.DataFrame, target_column: str) -> Dict:
    """
    Get class distribution for a target column
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target column
    
    Returns:
    --------
    distribution : Dict
        Dictionary containing class distribution info
    """
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    value_counts = data[target_column].value_counts()
    total = len(data)
    
    distribution = {
        'counts': value_counts.to_dict(),
        'percentages': {k: (v/total)*100 for k, v in value_counts.items()},
        'total': total,
        'n_classes': len(value_counts),
        'imbalance_ratio': value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1.0
    }
    
    return distribution


def check_imbalance(data: pd.DataFrame, target_column: str, threshold: float = 1.5) -> Dict:
    """
    Check if dataset is imbalanced
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target column
    threshold : float
        Imbalance ratio threshold (default: 1.5)
    
    Returns:
    --------
    result : Dict
        Dictionary containing imbalance check results
    """
    
    dist = get_class_distribution(data, target_column)
    
    is_imbalanced = dist['imbalance_ratio'] > threshold
    
    result = {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': dist['imbalance_ratio'],
        'threshold': threshold,
        'distribution': dist,
        'recommendation': _get_balancing_recommendation(dist['imbalance_ratio'])
    }
    
    return result


def _get_balancing_recommendation(imbalance_ratio: float) -> str:
    """Get balancing method recommendation based on imbalance ratio"""
    
    if imbalance_ratio < 1.5:
        return "No Balancing - Dataset is relatively balanced"
    elif imbalance_ratio < 3:
        return "SMOTE or Random Over-sampling - Mild imbalance"
    elif imbalance_ratio < 10:
        return "SMOTE recommended - Moderate imbalance"
    else:
        return "SMOTE + Random Under-sampling - Severe imbalance"
