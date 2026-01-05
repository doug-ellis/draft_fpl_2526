from wrangle_data_funcs import *
from modelling_funcs import *
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def get_training_df(training_years, n_gws, avg_type, alpha=0.3, rolling_gws=4):
    training_dfs = []
    for year in training_years:
        if avg_type=='ewma':
            gw_df = get_ewma_df(year, n_gws, alpha)
        elif avg_type=='rolling':
            gw_df = get_rolling_df(year, n_gws, rolling_gws)
        training_df = lag_data_for_training(gw_df).dropna(subset=['total_points_nw'])
        non_zero_players = training_df.groupby('full_name').sum().query('total_points_nw>0').index
        training_df_f = training_df.query('gw>10 and full_name in @non_zero_players')
        training_dfs.append(training_df_f.assign(year=year+2000))
    training_df_f = pd.concat(training_dfs, ignore_index=True)
    return training_df_f

def get_prediction_df(year, gw, avg_type, alpha=0.3, rolling_gws=4):
    if avg_type=='ewma':
        prediction_df = get_ewma_df(year, gw-1, alpha).drop(['ewma_team_goals_nw_opponent', 'ewma_team_points_nw_opponent'], axis=1)
    elif avg_type=='rolling':
        prediction_df = get_rolling_df(year, gw-1, rolling_gws).drop(['ewma_team_goals_nw_opponent', 'ewma_team_points_nw_opponent'], axis=1)

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

def test_model(training_df_f, features, model_func):
    _, rmse_dict = create_model(training_df_f, features, model_func, test=True)
    print(rmse_dict)
    return rmse_dict

def train_full_model(training_df, features, prediction_df, model_func):
    model_dict, _ = create_model(training_df, features, model_func, test=False)
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
    training_years = [23, 24, 25]
    training_n_gws = 38
    pred_year = 26
    pred_gw = 20
    alpha = 0.3
    rolling_gws = 4
    # features = [
    #     'assists', 'bonus', 'bps', 'clean_sheets', 'goals_conceded',
    #     'goals_scored', 'influence', 'creativity', 'threat', 'ict_index',
    #     'minutes', 'ewma_total_points', 
    #     'ewma_team_goals', 
    #     'ewma_team_points',
    #     'ewma_team_goals_nw_opponent', 
    #     'ewma_team_points_nw_opponent'
    #     ] + ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded', 
    #          'yellow_cards', 'saves']
    features = ['xP', 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
       'expected_assists', 'expected_goal_involvements', 'expected_goals',
       'expected_goals_conceded', 'goals_conceded', 'goals_scored',
       'ict_index', 'influence', 'minutes', 'own_goals', 'penalties_missed',
       'penalties_saved', 'red_cards', 'saves', 'starts', 'threat',
       'ewma_total_points', 
    #    'transfers_balance', 'transfers_in','transfers_out', 
       'value', 'yellow_cards',
       'ewma_team_goals', 'ewma_team_points', 'ewma_team_goals_nw_opponent',
       'ewma_team_points_nw_opponent', 
    #    'total_points'
       ]
    model_func = ElasticNet
    avg_type = 'rolling'
    output = f'transfer/outputs/predicted_gw{pred_gw}'
    return training_years, training_n_gws, pred_year, pred_gw, alpha, rolling_gws, features, model_func, avg_type, output

def main():
    training_years, training_n_gws, pred_year, pred_gw, alpha, rolling_gws, features, model_func, avg_type, output = get_params()
    # print(training_year, training_n_gws, pred_year, pred_gw, alpha, features, model, output)
    training_df = get_training_df(training_years, training_n_gws, avg_type, alpha, rolling_gws)
    training_df_scaled, _ = scale_df(training_df, features)
    _ = test_model(training_df_scaled, features, model_func)
    prediction_df = get_prediction_df(pred_year, pred_gw, avg_type, alpha, rolling_gws)
    prediction_df_scaled, _ = scale_df(prediction_df, features)
    pred_df = train_full_model(training_df_scaled, features, prediction_df_scaled, model_func)
    pred_df = merge_ownership_data(pred_df)
    pred_df_simple = pred_df[['full_name', 'position', 'team', 'ewma_total_points', 'predicted_points', 'owner']]
    pred_df.to_csv(f"{output}.csv", index=False)
    pred_df_simple.to_csv(f'{output}_simple.csv', index=False)

if __name__ == "__main__":
    main()