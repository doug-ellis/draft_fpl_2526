import pandas as pd
import unidecode
from useful_funcs import *

# Wrangle data
def import_data_from_vaastav(year, n_gws):
    year_range = f'20{year-1}-{year}'
    gw_df_list = []
    for i in range(1, n_gws+1):
        gw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{year_range}/gws/gw{i}.csv"
        gw_df = pd.read_csv(gw_url, index_col=0)
        gw_df['gw'] = i
        gw_df_list.append(gw_df)

    gw_df = pd.concat(gw_df_list)
    return gw_df.reset_index()

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




# Train model
# Test model
# Train model full
# Make predictions
