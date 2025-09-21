import pandas as pd
import unidecode

def combine_names(first_name, second_name):
    full_name = first_name + '_' + second_name    
    return full_name

def clean_name(name):
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = unidecode.unidecode(name)
    return name.strip().lower()

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



def clean_name2(name, separator):
    name = name.replace(" ", separator)
    name = name.replace("-", separator)
    name = unidecode.unidecode(name)
    return name.strip().lower()