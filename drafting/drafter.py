import pandas as pd 
from drafting_funcs import *

user_player = 3
nteams = 6

user_teams_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
pick_order = ([i+1 for i in range(nteams)] + [nteams-i for i in range(nteams)])*7+[i+1 for i in range(nteams)]

main_df = pd.read_csv('player_score_predictions_delta_manAdj.csv', index_col='full_name')
main_df['picked_by'] = None

for i in range(len(pick_order)):
    user_id = pick_order[i]

    # Calculate npicks before next pick (set to 0 if last pick)
    if i>len(pick_order) - nteams -1:
        npicks_before_next = 0
    else:
        next_pick = pick_order.index(user_id, i+1, len(pick_order))
        npicks_before_next = (next_pick - i)-1

    eligible_players = get_eligible_players(user_id, main_df)

    ##############
    choices = get_choices(user_id, eligible_players, main_df)
    user_pos_counts = main_df.query('picked_by == @user_id')['pos'].value_counts()
    picks_remaining_for11 = 11 - len(user_teams_dict[user_id])
    
    if len(user_teams_dict[user_id]) <11:
        allowed_positions = get_positions_needed_in_formation(user_pos_counts, picks_remaining_for11)
        choices = choices.query('pos in @allowed_positions')
    elif len(user_teams_dict[user_id]) >= 11:
        choices = choices.copy()
    ############

    if user_id ==user_player:
        print(f"It's your turn, recommended choices is {choose_player2(choices, npicks_before_next)}")
        response = input("Do you want to pick this player? (y/n): ").strip().lower()
        if response == 'y':
            # pick = choices.index[0]
            pick = choose_player2(choices, npicks_before_next)
        elif response == 'n':
            print("Please choose a player from the list of choices:")
            print(choices.index.tolist())
            pick = input("What player did you pick?").strip()
            while pick not in choices.index:
                print("Invalid choice. Please choose a player from the list of choices:")
                print(choices.index.tolist())
                pick = input("What player did you pick?").strip()
        elif response == '':
            pick = choose_player2(choices, npicks_before_next)
            
    else:
        print(f"Likely choices are: {main_df.query('is_available == True').sort_values('y_pred', ascending=False).index[:20].tolist()}")
        response = input("What player did they pick?").strip()
        if response == '':
            pick = choose_player2(choices, npicks_before_next)
        else:
            pick = response
        while pick not in main_df.index:
            print("Invalid choice. Please choose a valid player:")
            pick = input("What player did they pick?").strip()
        #### Make it so you have to pick a real player
    main_df, user_teams_dict = record_choice(user_teams_dict, user_id, pick, main_df, i)
    
    print(f'User {user_id} picks {pick} at pick {i + 1}')

# pd.DataFrame(user_teams_dict).T.to_csv('output/user_teams.csv')
main_df.to_csv('output/drafted_players.csv')