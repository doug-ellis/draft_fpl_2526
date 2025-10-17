import requests
import pandas as pd
from wrangle_data_funcs import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, rmse, r2

def create_model(training_df, features, model_func, test):
    model_dict = {}
    rmse_dict = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        # model = model_func()
        if model_func == XGBRegressor:
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif model_func == LinearRegression:
            model = LinearRegression()
        elif model_func == Ridge:
            model = Ridge(alpha=1.0)
        elif model_func == Lasso:
            model = Lasso(alpha=0.1)
        elif model_func == ElasticNet:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        else:
            raise ValueError("Unsupported model function")
        training_df_pos = training_df.query('position==@pos')
        X = training_df_pos[features]
        y = training_df_pos['total_points_nw']

        if test is True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model_dict[pos] = model.fit(X_train, y_train)
            mae, rmse, r2 = evaluate_model(X_test, y_test, model_dict[pos])
            rmse_dict[pos] = round(rmse, 3)
        else:
            model_dict[pos] = model.fit(X, y)
            rmse_dict[pos] = None
    return model_dict, rmse_dict

def predict_scores(prediction_df, features, model_dict):
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        prediction_df_pos = prediction_df.query('position==@pos')
        X_pred = prediction_df_pos[features]
        prediction_df.loc[prediction_df['position']==pos, 'predicted_points'] = model_dict[pos].predict(X_pred)
    return prediction_df