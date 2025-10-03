from itertools import product
import pandas as pd

def get_best_11(pred_df):
    # Possible formation constraints
    gk = 1
    def_range = range(3, 6)   # 3 to 5 DEF
    mid_range = range(3, 6)   # 3 to 5 MID
    fwd_range = range(1, 4)   # 1 to 3 FWD

    best_total = -float('inf')
    best_formation = None
    best_team = None

    for d, m, f in product(def_range, mid_range, fwd_range):
        if gk + d + m + f == 11:
            formation = {'GK': gk, 'DEF': d, 'MID': m, 'FWD': f}
            team = []
            for pos, n in formation.items():
                team.extend(pred_df.query(f'position == "{pos}"').head(n).to_dict('records'))
            total_pred = sum(player['predicted_points'] for player in team)
            if total_pred > best_total:
                best_total = total_pred
                best_formation = formation
                best_team = team

    best_11_df = pd.DataFrame(best_team)
    return best_formation, best_11_df.sort_values('predicted_points', ascending=False)

def get_owner_dict():
    owner_dict = {88376.0: 'Doug',
              93330.0: 'Marcus',
              87737.0: 'Will',
              85076.0: 'Lucas',
              85646.0: 'Dave',
              94654.0: 'Rory'  
              }
    return owner_dict

def get_all_best_11s(pred_df):
    owner_dict = get_owner_dict()
    best_11_list = []
    pred_points_list = []
    for owner_id, owner_name in owner_dict.items():
        pred_df_owner = pred_df.query(f'owner=={owner_id}')
        formation, best_11_df = get_best_11(pred_df_owner)
        best_11_df['owner'] = owner_name
        best_11_df['formation'] = f"{formation['DEF']}-{formation['MID']}-{formation['FWD']}"
        best_11_list.append(best_11_df)
        print(f"{owner_name}: predicted points {best_11_df['predicted_points'].sum().round(2)}")
        pred_points_list.append(best_11_df['predicted_points'].sum().round(2))
    return pred_points_list
    # all_best_11s_df = pd.concat(best_11_list).reset_index(drop=True)
    # return all_best_11s_df