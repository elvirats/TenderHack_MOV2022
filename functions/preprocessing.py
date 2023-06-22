from typing import List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from functions.word_preprocessing import words2vectors, code2words
from functions.utils import *
import pandas as pd

from pandarallel import pandarallel
pandarallel.initialize()


@dataclass
class TrainValData:
    """Store the train and valid data in specified dataclasses"""
    train: pd.DataFrame
    valid: pd.DataFrame


def unite_cols(data: pd.DataFrame, col1: str, col2: str) -> pd.Series:
    """Unite col1 and col2 columns into 'code' and drop the rest"""
    data['code'] = data[col1].combine_first(data[col2])
    data.drop([col1, col2], axis=1, inplace=True)
    return data

def date2features(data: pd.DataFrame, time_col: str = 'Дата'):
    """Append datetime features to dataframe"""
    data['time'] = pd.to_datetime(data[time_col])
    data['hour'] = data['time'].dt.hour.astype(object)
    data['minute'] = data['time'].dt.minute.astype(object)
    data['day'] = data['time'].dt.day.astype(object)
    data['day_of_week'] = data['time'].dt.day_of_week.astype(object)
    data['month'] = data['time'].dt.month.astype(object)
    data['quarter'] = data['time'].dt.quarter.astype(object)
    data['year'] = data['time'].dt.year.astype(object)
    data.drop(time_col, axis=1, inplace=True)
    data.drop('time', axis=1, inplace=True)
    return data

def preprocess_data(data: pd.DataFrame, extract_datetime_features: bool, vectorize_features: bool) -> pd.DataFrame:
    """Unite classifier columns in one and append datetime features"""
    data = unite_cols(data, 'ОКПД 2', 'КПГЗ')
    if extract_datetime_features:
        data = date2features(data)
    if vectorize_features:
        print('[INFO] Loading classifier database...')
        code_base = load_classifier_database()
        print('[INFO] Starting code to words process...')
        code_names = code2words(data['code'], code_base)
        print('[INFO] Transform words to vectors...')
        code_vector = words2vectors(code_names)
        ks_names_vector = words2vectors(data['Наименование КС'])
        print('[INFO] Unite vectors...')
        vector = code_vector + ks_names_vector
        data = pd.concat([data.reset_index(drop=True), vector], axis=1)
        data.drop(['code', 'Наименование КС'], axis=1, inplace=True)
    return data

def get_train_val_data_for_catboost(data: pd.DataFrame, 
                                    test_size=0.2,
                                    use_date_features: bool = False,
                                    vectorize_features: bool = False,
                                    status_columns: List[str] = ['Завершена']):
    """
    Return preprocessed X_train, X_val, y_train, y_val and scaler for inverse transform
    
    Steps:
    1. Filter out only specified status columns
    2. Calculate target drawdown
    3. Apply preprocessing to data
    4. Form feature and target data
    5. Perform train / val splitting
    6. Return feature and target data
    """
    # Filter out specified status columns
    data = data[data['Статус'].isin(status_columns)].reset_index(drop=True)
    data['Процент падения'] = data.apply(lambda x: apply_price_drawdown(x, 'НМЦК', 'Итоговая цена'), axis=1)
    
    # Data preprocessing
    data = preprocess_data(
                data, 
                extract_datetime_features=use_date_features, 
                vectorize_features=vectorize_features)

    print('[INFO] X y split...')
    # Split on features and target variables
    X = data.drop(['id', 'Статус', 'Итоговая цена', 'Участники', 'Ставки', 'Процент падения'], axis=1)
    y = data[['Участники', 'Процент падения']]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    features = TrainValData(train=X_train, valid=X_val)
    drawdown_target = TrainValData(train=y_train['Процент падения'], valid=y_val['Процент падения'])
    num_competitors_target = TrainValData(train=y_train['Участники'], valid=y_val['Участники'])

    print('[INFO] Done...')
    return features, drawdown_target, num_competitors_target
