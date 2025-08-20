import requests, json
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
from useful_funcs import *


def get_latest_players_df():
    base_url = 'https://fantasy.premierleague.com/api/'
    r = requests.get(base_url+'bootstrap-static/').json()
    players = pd.json_normalize(r['elements'])
    return players


def clean_simplify_df(players_df):
    players_df['full_name'] = players_df.apply(lambda row: combine_clean_names(row['first_name'], row['second_name']), axis=1)
    players_df = players_df.set_index('full_name')

    players_simple = players_df[['element_type', 'now_cost', 'team', 'team_code', 'total_points', 'team_join_date']]

    pos_dict = {1: 'GK',
            2: 'DEF',
            3: 'MID',
            4: 'FWD'}
    players_simple['pos'] = players_simple['element_type'].map(pos_dict)
    return players_simple

def add_team_goals(players_simple):
    teams_url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/2024-25/teams.csv'
    teams = pd.read_csv(teams_url)
    teams = teams.rename(columns={'code': 'team_code', 
                                'name': 'team'})

    teamcode_dict = dict(zip(teams['team_code'], teams['team']))
    teamcode_dict[56] = 'Sunderland'
    teamcode_dict[90] = 'Burnley'
    teamcode_dict[2] = 'Leeds'

    players_simple['team_name'] = players_simple['team_code'].map(teamcode_dict)

    pl2425_goals = pd.read_csv('pl_2425_goalsbyteam.csv')
    pl2425_goals = pl2425_goals.rename(columns={'team':'team_name'})

    players_simple2 = players_simple.reset_index(
                        ).merge(pl2425_goals, on='team_name', how='left'
                                ).set_index('full_name')
    
    players_simple2['GF'] = players_simple2['GF'].fillna('37.7') # Filling promoted teams goals scored and conceded with average from last 10 years
    players_simple2['GA'] = players_simple2['GA'].fillna('67.16666667')

    cleaned_players_df = players_simple2.drop(['element_type', 'team', 'team_code'], axis=1)
    return cleaned_players_df

def train_model(players_cleaned):
    predictions = []
    for pos in players_cleaned['pos'].unique():
        # print(pos)
        df = players_cleaned.query('pos==@pos')
        X_full = df[['now_cost', 'GF', 'GA']]
        y_full = df['total_points']

        # Training the model on scoring players only
        df2 = df.query('total_points>0')
        X_train = df2[['now_cost', 'GF', 'GA']]
        y_train = df2['total_points']

        # Fitting model
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        # Actually fitting the model to all players
        y_pred_full = reg.predict(X_full)
        prediction = y_full.to_frame()
        prediction['y_pred'] = y_pred_full

        predictions.append(prediction)

    pred_df = pd.concat(predictions)
    return pred_df

def save_dfs(dirty_players_df, cleaned_players_df, pred_df):
    date = datetime.datetime.now().date().isoformat()
    dirty_players_df.to_csv(f'dirty_player_csvs/players_dirty_{date}.csv')
    cleaned_players_df.to_csv(f'cleaned_player_csvs/players_clean_{date}.csv')
    pred_df.to_csv(f'predicted_scores_csvs/players_scores_{date}.csv')

def main():
    dirty_players_df = get_latest_players_df()
    cleaned_players_df = clean_simplify_df(dirty_players_df)
    players_with_goals = add_team_goals(cleaned_players_df)
    pred_df = train_model(players_with_goals)
    save_dfs(dirty_players_df, cleaned_players_df, pred_df)

if __name__ == "__main__":
    main()



# Need to introduce logic to map date onto gw (or maybe it's in the df somewhere) and save the dataframe only if gw has changed)