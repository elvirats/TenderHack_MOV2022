from typing import List
from catboost import cv
from catboost import Pool
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd


def train_and_validate_catboost_cv(features: pd.DataFrame, target: pd.Series):
    """Train and return the result"""
    cat_features = features.select_dtypes('object').columns.to_list()
    params = {
        'loss_function': 'MAE',
        'iterations': 1000,
        'custom_loss': 'MAE',
        'random_seed': 42,
        'learning_rate': 0.1
    }

    result = cv(
        params=params,
        pool=Pool(features, label=target, cat_features=cat_features),
        fold_count=5,
        shuffle=True,
        partition_random_seed=0,
        plot=True,
        verbose=1
    )
    return result

def eval_model(model: CatBoostRegressor, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Evaluate the model and return the R2 and MAE scores"""
    predictions = model.predict(X_val)
    r2 = r2_score(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)
    return {'R2': r2, 'MAE': mae}

def train_and_validate_catboost(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                y_train: pd.DataFrame, y_val: pd.DataFrame,
                                loss_function: str = 'MAE', custom_metric: str = 'MAE', 
                                iterations: int = 300, lr: float = 0.1,
                                verbose: int = 1, show_score: bool = True, 
                                use_text_features: bool = False, text_features: List[str] = ['Наименование КС'],
                                use_gpu: bool=True):
    """Fit model on train data and return the model and the score for validation data"""
    # if (not use_text_features) and text_features:
    #     raise AttributeError("either pass in text features or turn off use_text_features argument")
    cat_features = X_train.select_dtypes('object').columns.to_list()

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=lr,
        loss_function=loss_function,
        custom_metric=custom_metric,
        task_type="GPU" if use_gpu else "CPU",
        devices='0:1'
    )
    model.fit(
        X_train, y_train, 
        cat_features=cat_features,
        text_features=text_features if use_text_features else None,
        verbose=verbose)
    scores = eval_model(model, X_val, y_val)
    if show_score:
        print(pd.DataFrame(scores, index=['Score']))
    return model, scores