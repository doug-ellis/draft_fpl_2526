from wrangle_data_funcs import *
from modelling_funcs import *
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def get_training_df(year, n_gws, alpha):
    gw_df = get_ewma_df(year, n_gws, alpha)
    training_df = lag_data_for_training(gw_df).dropna(subset=['total_points_nw'])
    non_zero_players = training_df.groupby('full_name').sum().query('total_points_nw>0').index
    training_df_f = training_df.query('gw>1 and full_name in @non_zero_players')
    return training_df_f

def get_prediction_df(year, gw, alpha):
    prediction_df = get_ewma_df(year, gw-1, alpha).drop(['ewma_team_goals_nw_opponent', 'ewma_team_points_nw_opponent'], axis=1)
    prediction_df = prediction_df.query(f'gw=={gw-1}')

    teamcode_dict = get_teamcodes(26)
    fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
    r = requests.get(fixtures_url).json()
    fixtures_df = pd.json_normalize(r)
    fixtures_df_gw = fixtures_df.query(f'event=={gw}')
    h_teams = fixtures_df_gw['team_h'].map(teamcode_dict)
    a_teams = fixtures_df_gw['team_a'].map(teamcode_dict)
    fixture_dict = {**dict(zip(h_teams, a_teams)), **dict(zip(a_teams, h_teams))}
    prediction_df['team_name_nw_opponent'] = prediction_df['team'].map(fixture_dict)
    opp_team_df = prediction_df[['team', 'ewma_team_goals', 'ewma_team_points']].groupby('team').first().reset_index()
    prediction_df = prediction_df.merge(opp_team_df, left_on='team_name_nw_opponent', right_on='team', suffixes=('', '_nw_opponent'))
    return prediction_df

def test_model(training_df_f, features, model):
    _, rmse_dict = create_model(training_df_f, features, model, test=True)
    print(rmse_dict)

def train_full_model(training_df, features, prediction_df, model):
    model_dict, _ = create_model(training_df, features, model, test=False)
    pred_df = predict_scores(prediction_df.dropna(), features, model_dict)
    return pred_df

def merge_ownership_data(pred_df):
    league_url = 'https://draft.premierleague.com/api/league/19188/element-status'
    r = requests.get(league_url).json()
    ownership_df = pd.json_normalize(r['element_status'])

    url = 'https://draft.premierleague.com/api/bootstrap-static'
    req = requests.get(url).json()
    players_df = pd.json_normalize(req['elements'])
    players_df['full_name'] = combine_names(players_df['first_name'], players_df['second_name']).apply(clean_name)
    merge_name_df = players_df[['id', 'full_name']]
    ownership_df_name = ownership_df.merge(merge_name_df, how='left', left_on='element', right_on='id')
    ownership_df_to_merge = ownership_df_name[['full_name', 'owner']]

    pred_df_owners = pred_df.merge(ownership_df_to_merge, on='full_name')
    return pred_df_owners

def get_params():
    training_year = 25
    training_n_gws = 38
    pred_year = 26
    pred_gw = 6
    alpha = 0.3
    features = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'goals_conceded',
        'goals_scored', 'influence', 'creativity', 'threat', 'ict_index',
        'minutes', 'ewma_total_points', 'ewma_team_goals', 'ewma_team_points',
        'ewma_team_goals_nw_opponent', 
        'ewma_team_points_nw_opponent'
        ]
    model = LinearRegression()
    output = f'transfer/outputs/predicted_gw{pred_gw}'
    return training_year, training_n_gws, pred_year, pred_gw, alpha, features, model, output

def main():
    training_year, training_n_gws, pred_year, pred_gw, alpha, features, model, output = get_params()
    training_df = get_training_df(training_year, training_n_gws, alpha)
    test_model(training_df, features, model)
    prediction_df = get_prediction_df(pred_year, pred_gw, alpha)
    pred_df = train_full_model(training_df, features, prediction_df, model)
    pred_df = merge_ownership_data(pred_df)
    pred_df_simple = pred_df[['full_name', 'position', 'team', 'ewma_total_points', 'predicted_points', 'owner']]
    pred_df.to_csv(f"{output}.csv", index=False)
    pred_df_simple.to_csv(f'{output}_simple.csv', index=False)

if __name__ == "__main__":
    main()