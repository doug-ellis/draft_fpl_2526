import pandas as pd
from wrangle_data import * 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_model(merged_ewma_df, feature_cols, target_col, model):   
    # Drop rows with NaN values in feature or target columns
    gw_df = merged_ewma_df.dropna(subset=feature_cols + [target_col])

    X = gw_df[feature_cols]
    y = gw_df[target_col]

    model.fit(X, y)
    return model

def test_model(merged_ewma_df, feature_cols, target_col, model):
    # Drop rows with NaN values in feature or target columns
    gw_df = merged_ewma_df.dropna(subset=feature_cols + [target_col])



    X = gw_df[feature_cols]
    y = gw_df[target_col]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return mae, mse, r2

def main():
    
