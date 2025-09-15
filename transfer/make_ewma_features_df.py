import pandas as pd 
from useful_funcs import *

def get_dfs(year):
    year_range = f'20{year-1}-{year}'
    gw_df_list = []
    for i in range(1, 39):
        gw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{year_range}/gws/gw{i}.csv"
        gw_df = pd.read_csv(gw_url, index_col=0)
        gw_df['gw'] = i
        gw_df_list.append(gw_df)

        gw_df = pd.concat(gw_df_list)
    return gw_df
    
def add_goals(gw_df):
    gw_df['team_goals'] = gw_df.apply(lambda row: get_team_goals(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['opponent_goals'] = gw_df.apply(lambda row: get_opponent_goals(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['team_points'] = gw_df.apply(lambda row: get_team_points(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['opponent_points'] = gw_df['team_points'].apply(get_opponent_points)

    return gw_df

def clean_name_in_index(gw_df):
    gw_df['full_name'] = [clean_name(idx) for idx in gw_df.index]
    return gw_df

def ewma(gw_df, groupby_col, value_cols, alpha, rename_dict, remerge_cols):
    # Ensure the DataFrame is sorted by 'full_name' and 'gw'
    gw_df_sorted = gw_df.sort_values([groupby_col, 'gw']).reset_index()

    # Apply EWMA within each group
    ewma_gw_df = (
        gw_df_sorted
        .groupby(groupby_col, group_keys=False)
        [value_cols]
        .apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    )
    ewma_gw_df.rename(columns=rename_dict, inplace=True)
    ewma_gw_df = gw_df_sorted[remerge_cols].join(ewma_gw_df)
    return ewma_gw_df

def merge_ewma(ewma_gw_df, gw_df, merge_cols):
    ewma_gw_df_merge = gw_df[merge_cols].join(ewma_gw_df)
    return ewma_gw_df_merge

def get_teams_df(gw_df):
    gw_df_teams = gw_df[['team', 'gw', 'team_goals']]
    gw_df_teams = gw_df_teams.groupby(['team', 'gw']).first().reset_index()
    return gw_df_teams

def get_teamcodes(year):
    teams_url = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/20{year-1}-{year}/teams.csv'
    teams = pd.read_csv(teams_url)
    teamcode_dict = dict(zip(teams['id'], teams['name']))
    return teamcode_dict

def lag_feature(ewma_gw_df, feature):
    new_feature = (
    ewma_gw_df
    .sort_values(['full_name', 'gw'])
    .groupby('full_name')[feature]
    .shift(-1))
    return new_feature
    
def merge_lag_player_ewma(ewma_gw_df_players, ewma_gw_df_teams, year):
    teamcode_dict = get_teamcodes(year)
    ewma_gw_df_players['opponent_team_name'] = ewma_gw_df_players['opponent_team'].map(teamcode_dict)
    ewma_gw_df_opponent_team = ewma_gw_df_teams.rename(columns={'team': 'opponent_team_name', 
                                                               'ewma_team_goals': 'ewma_nw_opponent_goals'})
    ewma_gw_df = ewma_gw_df_players.merge(ewma_gw_df_teams, on=['team', 'gw'], how='left')
    ewma_gw_df['nw_total_points'] = lag_feature(ewma_gw_df, 'total_points')
    ewma_gw_df['nw_opponent'] = lag_feature(ewma_gw_df, 'opponent_team_name')
    ewma_gw_df = ewma_gw_df.merge(ewma_gw_df_opponent_team, on=['opponent_team_name', 'gw'], how='left')
    return ewma_gw_df

def make_ewma_features_df(gw_df, year, alpha):
    # gw_df = get_dfs(year)
    gw_df = add_goals(gw_df)
    gw_df = clean_name_in_index(gw_df)

    player_value_cols = ['assists', 'bonus', 'bps', 'clean_sheets', 'goals_conceded', 'goals_scored',
                      'influence', 'creativity', 'threat', 'ict_index', 'minutes', 'total_points']
    merge_cols_players = ['full_name', 'gw', 'total_points', 'position','team','opponent_team']
    ewma_gw_df_players = ewma(gw_df, 'full_name', player_value_cols, alpha, {'total_points': 'ewma_total_points'}, merge_cols_players)

    gw_df_teams = get_teams_df(gw_df)
    team_value_cols = ['team_goals']
    merge_cols_teams = ['team', 'gw']
    ewma_gw_df_teams = ewma(gw_df_teams, 'team', team_value_cols, alpha, {'team_goals': 'ewma_team_goals'}, merge_cols_teams)

    ewma_merge_lag = merge_lag_player_ewma(ewma_gw_df_players, ewma_gw_df_teams, year)
    return ewma_merge_lag