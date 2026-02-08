import pandas as pd
import requests
import unidecode
from sklearn.preprocessing import StandardScaler

def import_data_from_vastaav(year, n_gws):
    year_range = f'20{year-1}-{year}'
    gw_df_list = []
    for i in range(1, n_gws+1):
        # gw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{year_range}/gws/gw{i}.csv"
        gw_url = f"https://raw.githubusercontent.com/doug-ellis/Fantasy-Premier-League/refs/heads/doug-ellis/gw7/data/{year_range}/gws/gw{i}.csv"
        
        gw_df = pd.read_csv(gw_url, index_col=0)
        gw_df['gw'] = i
        gw_df_list.append(gw_df)

    gw_df = pd.concat(gw_df_list)
    return gw_df.reset_index()

def get_team_goals(was_home, team_h_score, team_a_score):
    if was_home:
        return team_h_score
    else:
        return team_a_score
def get_opponent_goals(was_home, team_h_score, team_a_score):
    if was_home:
        return team_a_score
    else:
        return team_h_score
def get_team_points(was_home, team_h_score, team_a_score):
    if was_home:
        return 3 if team_h_score > team_a_score else 0 if team_h_score < team_a_score else 1
    else:
        return 3 if team_a_score > team_h_score else 0 if team_a_score < team_h_score else 1
def get_opponent_points(team_points):
    if team_points == 3:
        return 0
    elif team_points == 0:
        return 3
    else:
        return 1

def add_team_data(gw_df):
    gw_df['team_goals'] = gw_df.apply(lambda row: get_team_goals(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['opponent_goals'] = gw_df.apply(lambda row: get_opponent_goals(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['team_points'] = gw_df.apply(lambda row: get_team_points(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['opponent_points'] = gw_df['team_points'].apply(get_opponent_points)
    return gw_df

def combine_names(first_name, second_name):
    full_name = first_name + '_' + second_name    
    return full_name

def clean_name(name):
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = unidecode.unidecode(name)
    return name.strip().lower()

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

def roll(gw_df, groupby_col, value_cols, rename_dict, remerge_cols, rolling_gws):
    # Ensure the DataFrame is sorted by 'full_name' and 'gw'
    gw_df_sorted = gw_df.sort_values([groupby_col, 'gw']).reset_index()

    # Apply rolling mean within each group
    rolling_gw_df = (
        gw_df_sorted
        .groupby(groupby_col, group_keys=False)
        [value_cols]
        .apply(lambda x: x.rolling(window=rolling_gws, min_periods=1).mean())
    )
    rolling_gw_df.rename(columns=rename_dict, inplace=True)
    rolling_gw_df = gw_df_sorted[remerge_cols].join(rolling_gw_df)
    return rolling_gw_df

def get_teams_df(gw_df):
    gw_df_teams = gw_df[['team', 'gw', 'team_goals', 'team_points']]
    gw_df_teams = gw_df_teams.groupby(['team', 'gw']).first().reset_index()
    return gw_df_teams

def get_teamcodes(year):
    teams_url = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/20{year-1}-{year}/teams.csv'
    teams = pd.read_csv(teams_url)
    teamcode_dict = dict(zip(teams['id'], teams['name']))
    return teamcode_dict

def merge_ewma_dfs(ewma_gw_df_players, ewma_gw_df_teams, year):
    teamcode_dict = get_teamcodes(year)
    ewma_gw_df_players['opponent_team_name'] = ewma_gw_df_players['opponent_team'].map(teamcode_dict)
    ewma_gw_df = ewma_gw_df_players.merge(ewma_gw_df_teams, on=['team', 'gw'], how='left')

    ewma_gw_df_opponent_team = ewma_gw_df_teams.rename(columns={'team': 'opponent_team_name', 
                                                            'ewma_team_goals': 'ewma_team_goals_nw_opponent',
                                                            'ewma_team_points': 'ewma_team_points_nw_opponent'})
    ewma_gw_df = ewma_gw_df.merge(ewma_gw_df_opponent_team, on=['opponent_team_name', 'gw'], how='left')
    return ewma_gw_df

def lag_feature(ewma_gw_df, feature):
    new_feature = (
    ewma_gw_df
    .sort_values(['full_name', 'gw'])
    .groupby('full_name')[feature]
    .shift(-1))
    return new_feature

def get_ewma_df(year, gw, ewma_alpha):
    gw_df = import_data_from_vastaav(year, gw)
    gw_df = add_team_data(gw_df)

    gw_df['full_name'] = gw_df['name'].apply(clean_name)

    player_value_cols = ['xP', 'assists', 'bonus', 'bps',
       'clean_sheets', 'creativity', 'expected_assists',
       'expected_goal_involvements', 'expected_goals',
       'expected_goals_conceded', 'goals_conceded', 'goals_scored',
       'ict_index', 'influence', 'minutes',
       'own_goals', 'penalties_missed', 'penalties_saved',
       'red_cards', 'saves', 'starts',
       'threat', 'total_points', 'transfers_balance',
       'transfers_in', 'transfers_out', 'value', 'yellow_cards']
    merge_cols_players = ['full_name', 'gw', 'total_points', 'position','team','opponent_team']
    ewma_gw_df_players = ewma(gw_df, 'full_name', player_value_cols, ewma_alpha, {'total_points': 'ewma_total_points'}, merge_cols_players)

    gw_df_teams = get_teams_df(gw_df)
    team_value_cols = ['team_goals', 'team_points']
    merge_cols_teams = ['team', 'gw']
    ewma_gw_df_teams = ewma(gw_df_teams, 'team', team_value_cols, ewma_alpha, {'team_goals': 'ewma_team_goals',
                                                                            'team_points': 'ewma_team_points'}, merge_cols_teams)
    merged_ewma_df = merge_ewma_dfs(ewma_gw_df_players, ewma_gw_df_teams, year)
    return merged_ewma_df

def get_rolling_df(year, gw, rolling_gws):
    gw_df = import_data_from_vastaav(year, gw)
    gw_df = add_team_data(gw_df)

    gw_df['full_name'] = gw_df['name'].apply(clean_name)

    player_value_cols = ['xP', 'assists', 'bonus', 'bps',
       'clean_sheets', 'creativity', 'expected_assists',
       'expected_goal_involvements', 'expected_goals',
       'expected_goals_conceded', 'goals_conceded', 'goals_scored',
       'ict_index', 'influence', 'minutes',
       'own_goals', 'penalties_missed', 'penalties_saved',
       'red_cards', 'saves', 'starts',
       'threat', 'total_points', 'transfers_balance',
       'transfers_in', 'transfers_out', 'value', 'yellow_cards']
    merge_cols_players = ['full_name', 'gw', 'total_points', 'position','team','opponent_team']
    ewma_gw_df_players = roll(gw_df, 'full_name', player_value_cols, {'total_points': 'ewma_total_points'}, merge_cols_players, rolling_gws)

    gw_df_teams = get_teams_df(gw_df)
    team_value_cols = ['team_goals', 'team_points']
    merge_cols_teams = ['team', 'gw']
    ewma_gw_df_teams = roll(gw_df_teams, 'team', team_value_cols, {'team_goals': 'ewma_team_goals',
                                                                            'team_points': 'ewma_team_points'}, merge_cols_teams, rolling_gws)
    merged_ewma_df = merge_ewma_dfs(ewma_gw_df_players, ewma_gw_df_teams, year)
    return merged_ewma_df

def lag_data_for_training(merged_ewma_df):
    merged_ewma_df['total_points_nw'] = lag_feature(merged_ewma_df, 'total_points')
    merged_ewma_df['opponent_nw'] = lag_feature(merged_ewma_df, 'opponent_team_name')
    return merged_ewma_df

def scale_df(df, features):
    scaler = StandardScaler()
    scaler.fit(df[features])
    df[features] = scaler.transform(df[features])
    return df, scaler

def get_fixture_dict(gw, year):
    teamcode_dict = get_teamcodes(year)
    fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
    r = requests.get(fixtures_url).json()
    fixtures_df = pd.json_normalize(r)
    fixtures_df_gw = fixtures_df.query(f'event=={gw}')
    h_teams = fixtures_df_gw['team_h'].map(teamcode_dict)
    a_teams = fixtures_df_gw['team_a'].map(teamcode_dict)
    fixture_dict = {**dict(zip(h_teams, a_teams)), **dict(zip(a_teams, h_teams))}
    return fixture_dict