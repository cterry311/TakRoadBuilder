import tak_engine


from engine_clients.roadbuilder import RoadBuilder




def get_model_move(client,game: tak_engine.Tak, num_positions, c_puct, urgency):
    tps = game.to_tps()
    best_move = client.best_move(tps, max_sims=num_positions, c_puct=c_puct, urgency=urgency, komi=2)[0]
    return best_move




def save_moves_to_ptn(game, moves, player_white, player_black, save_path):
    ptn_game = f'[size "6"]\n[player1 "{player_white}"]\n[player2 "{player_black}"]\n[komi "2"]\n'

    for i, move in enumerate(moves):
        if i % 2 == 0:
            ptn_game += f'{(i // 2 ) + 1}. '
        ptn_game += move + ' '
        if i % 2 == 1:
            ptn_game += '\n'

    with open(save_path, 'w') as f:
        f.write(ptn_game)


def run_self_games(model1, model2, num_positions, c_puct, urgency, num_games, komi, save_games=False):
    model_client1 = RoadBuilder(model1)
    model_client2 = RoadBuilder(model2)
    name_1 = model1.split('/')[-1].split('.')[0]
    name_2 = model2.split('/')[-1].split('.')[0]
    wins = 0
    draws = 0
    losses = 0
    for i in range(num_games):
        game = tak_engine.Tak(komi=komi)
        moves = []
        k = i
        while True:
            if k % 2 == 0:
                move = get_model_move(model_client1, game, num_positions, c_puct, urgency)
                moves.append(move)
                game.make_move(move)
                if game.result() != 0:
                    break
            else:
                move = get_model_move(model_client2, game, num_positions, c_puct, urgency)
                moves.append(move)
                game.make_move(move)
                if game.result() != 0:
                    break
            k += 1
        game_result = game.result()
        if game_result == 3:
            draws += 1
        elif i % 2 == 0:
            if game_result == 1:
                wins += 1
            elif game_result == 2:
                losses += 1
            else:
                raise ValueError(f'game result {game_result} not recognized')
        elif i % 2 == 1:
            if game_result == 2:
                wins += 1
            elif game_result == 1:
                losses += 1
            else:
                raise ValueError(f'game result {game_result} not recognized')
        else:
            raise ValueError(f'game result {game_result} not recognized')
        if save_games:
            player_white = name_1 if i % 2 == 0 else name_2
            player_black = name_1 if i % 2 == 1 else name_2
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/{name_1}___{name_2}___{i}_tiltak_trained.ptn')
    print(f'wins: {wins}, losses: {losses}, draws: {draws}')
    print(f'total games: {wins + losses + draws}')
    return wins, losses, draws


# print('self games, model 9_34')
# run_self_games('model_checkpoints/version_9_34.pt', 'model_checkpoints/version_9_5.pt', 20_000, 1.5, -0.2, 100, 2, save_games=True)
# the version 9_34 is preforming much worse than 9_5 when given deeper searches, consider retraining on more search depth




# print('self games, model 9_51')
# run_self_games('model_checkpoints/version_9_51.pt', 'model_checkpoints/version_9_5.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 80, losses: 16, draws: 4
# total games: 100


# print('self games, model 9_51')
# run_self_games('model_checkpoints/version_9_51.pt', 'model_checkpoints/version_9_5.pt', 20_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 50, losses: 35, draws: 15
# total games: 100


# print('self games, model 9_51 vs 9_34')
# run_self_games('model_checkpoints/version_9_51.pt', 'model_checkpoints/version_9_34.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 27, losses: 62, draws: 11
# total games: 100


# print('self games, model 9_51 vs 9_34')
# run_self_games('model_checkpoints/version_9_51.pt', 'model_checkpoints/version_9_34.pt', 20_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 10, losses: 87, draws: 3
# total games: 100


# print('self games, model 10_16 vs 9_34')
# run_self_games('model_checkpoints/version_10_16.pt', 'model_checkpoints/version_9_34.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 38, losses: 45, draws: 17
# total games: 100


# print('self games, model 10_16 vs 9_34')
# run_self_games('model_checkpoints/version_10_16.pt', 'model_checkpoints/version_9_34.pt', 20_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 12, losses: 80, draws: 8
# total games: 100


# print('self games, model 11_25 vs 9_34')
# run_self_games('model_checkpoints/version_11_25.pt', 'model_checkpoints/version_9_34.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 26, losses: 49, draws: 25
# total games: 100



# print('self games, model 11_10 vs 9_34')
# run_self_games('model_checkpoints/version_11_10.pt', 'model_checkpoints/version_9_34.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 35, losses: 52, draws: 13
# total games: 100


# print('self games, model 12_10 vs 9_34')
# run_self_games('model_checkpoints/version_12_10.pt', 'model_checkpoints/version_9_34.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 32, losses: 51, draws: 17
# total games: 100


# print('self games, model 13_5 vs 9_34')
# run_self_games('model_checkpoints/version_13_5.pt', 'model_checkpoints/version_9_34.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)
# wins: 34, losses: 50, draws: 16
# total games: 100


print('self games, model 13_0 vs 9_34')
run_self_games('model_checkpoints/version_13_0.pt', 'model_checkpoints/version_9_34.pt', 5_000, 1.5, -0.2, 100, 2, save_games=True)