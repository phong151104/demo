"""
Data Processing Module
Các chức năng xử lý và tiền xử lý dữ liệu
"""

from .outlier_handler import (
    OutlierHandler,
    handle_outliers
)

from .encoder import (
    CategoricalEncoder,
    encode_categorical,
    recommend_encoding
)

from .preprocessing_pipeline import (
    PreprocessingPipeline,
    create_pipeline
)

__all__ = [
    'OutlierHandler',
    'handle_outliers',
    'CategoricalEncoder',
    'encode_categorical',
    'recommend_encoding',
    'PreprocessingPipeline',
    'create_pipeline'
]

# TODO: Implement additional data processing functions
# - handle_missing_values()
# - scale_features()
# - balance_data()
# - create_bins()

