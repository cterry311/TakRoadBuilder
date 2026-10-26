from engine_clients.beginner_bot import BeginnerBotEngine
from engine_clients.taktician_bot import TakticianTEI
from engine_clients.tiltak_bot import Tiltak

import torch
from model.tak_net import TakNet

import tak_engine



info = torch.load(f'model_checkpoints/version_6_1.pt')
model = TakNet(**info['params'])
model.load_state_dict(info['weights'])


def get_model_move(tak: tak_engine.Tak, model, depth):
    if depth == -1:
        filtered_moves = tak.legal_moves_raw()
        value = 0
    else:
        filtered_moves, value = tak.filtered_moves_raw(depth, True)
    if value == 1:
        return tak.raw_to_ptn(filtered_moves[0])
    model.eval()
    with torch.no_grad():
        tensor_rep = torch.from_numpy(tak.to_tensor())
        legal_mask = torch.from_numpy(tak.get_legal_mask())
        tensor_rep = tensor_rep.reshape(1, 25, 6, 6)
        policy, _ = model(tensor_rep)
        masked_output = policy.flatten().masked_fill(~(legal_mask.flatten().bool()), -1e9)
        index_order = torch.argsort(masked_output, descending=True)
        selected_move = None
        for i in range(index_order.shape[0]):
            move_at_i = tak_engine.index_to_raw(index_order[i].item())
            if move_at_i in filtered_moves:
                selected_move = tak.raw_to_ptn(move_at_i)
                break
    return selected_move



# moves = []
# ptn_game = "[size \"6\"]\n"
# i = 1
# while True:
#     ptn_game += f'{i}. '
#     engine_move = bot.best_move(moves=moves)['move']
#     print(f'engine_move {engine_move}')
#     game.make_move(engine_move)
#     moves.append(engine_move)
#     ptn_game += engine_move
#     if game.result() != 0:
#         print(f'game result: {game.result()}')
#         break
#     move = get_model_move(game, model, -1)
#     print(f'model_move {move}')
#     game.make_move(move)
#     moves.append(move)
#     ptn_game += ' ' + move +'\n'
#     if game.result() != 0:
#         print(f'game result: {game.result()}')
#         break
#     i += 1
#
# with open('test2.ptn', 'w') as f:
#     f.write(ptn_game)

def save_moves_to_ptn(game, moves, player_white, player_black, save_path):
    ptn_game = f'[size "6"]\n[player1 "{player_white}"]\n[player2 "{player_black}"]\n'

    for i, move in enumerate(moves):
        if i % 2 == 0:
            ptn_game += f'{(i // 2 ) + 1}. '
        ptn_game += move + ' '
        if i % 2 == 1:
            ptn_game += '\n'

    with open(save_path, 'w') as f:
        f.write(ptn_game)


def run_beginner_bot_games(search_depth, beginner_bot_time, num_games, save_games=False):
    bot = BeginnerBotEngine()
    wins = 0
    draws = 0
    losses = 0
    for i in range(num_games):
        game = tak_engine.Tak()
        moves = []
        k = i
        while True:
            if k % 2 == 0:
                move = get_model_move(game, model, search_depth)
                moves.append(move)
                game.make_move(move)
                if game.result() != 0:
                    break
            else:
                move = bot.best_move(moves=moves, size=6, time_ms=beginner_bot_time)['move']
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
            player_white = 'Model' if i % 2 == 0 else 'BeginnerBot'
            player_black = 'Model' if i % 2 == 1 else 'BeginnerBot'
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/game_beginnerBot_{i}.ptn')
    print(f'wins: {wins}, losses: {losses}, draws: {draws}')
    print(f'total games: {wins + losses + draws}')
    return wins, losses, draws


def run_taktician_bot_games(search_depth, taktician_bot_time, num_games, save_games=False):
    bot = TakticianTEI()
    wins = 0
    draws = 0
    losses = 0
    for i in range(num_games):
        game = tak_engine.Tak()
        moves = []
        k = i
        while True:
            if k % 2 == 0:
                move = get_model_move(game, model, search_depth)
                moves.append(move)
                game.make_move(move)
                if game.result() != 0:
                    break
            else:
                move = bot.best_move(moves=moves, movetime_ms=taktician_bot_time)
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
            player_white = 'Model' if i % 2 == 0 else 'Taktician'
            player_black = 'Model' if i % 2 == 1 else 'Taktician'
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/game_tactician_{i}.ptn')
    print(f'wins: {wins}, losses: {losses}, draws: {draws}')
    print(f'total games: {wins + losses + draws}')
    return wins, losses, draws


def run_tiltak_bot_games(search_depth, taktician_bot_time, num_games, save_games=False):
    bot = Tiltak()
    wins = 0
    draws = 0
    losses = 0
    for i in range(num_games):
        game = tak_engine.Tak()
        moves = []
        k = i
        while True:
            if k % 2 == 0:
                move = get_model_move(game, model, search_depth)
                moves.append(move)
                game.make_move(move)
                if game.result() != 0:
                    break
            else:
                move = bot.best_move(moves=moves, movetime_ms=taktician_bot_time)
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
            player_white = 'Model' if i % 2 == 0 else 'Tiltak'
            player_black = 'Model' if i % 2 == 1 else 'Tiltak'
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/game_tiltak_{i}_4_2.ptn')
    print(f'wins: {wins}, losses: {losses}, draws: {draws}')
    print(f'total games: {wins + losses + draws}')
    return wins, losses, draws



# print('\ndepth 2, 100ms')
# run_beginner_bot_games(2, 100, 100)
# print('\ndepth 2, 500ms')
# run_beginner_bot_games(2, 500, 100)
# print('\ndepth 2, 1000ms')
# run_beginner_bot_games(2, 1000, 100)
# print('\ndepth 2, 2000ms')
# run_beginner_bot_games(2, 2000, 100)
#
# print('\n\n\ndepth 3, 100ms')
# run_beginner_bot_games(3, 100, 100)
# print('\ndepth 3, 500ms')
# run_beginner_bot_games(3, 500, 100)
# print('\ndepth 3, 1000ms')
# run_beginner_bot_games(3, 1000, 100)
# print('\ndepth 3, 2000ms')
# run_beginner_bot_games(3, 2000, 100)



# depth 2, 100ms
# wins: 90, losses: 9, draws: 1
# total games: 100
#
# depth 2, 500ms
# wins: 78, losses: 18, draws: 4
# total games: 100
#
# depth 2, 1000ms
# wins: 71, losses: 28, draws: 1
# total games: 100
#
# depth 2, 2000ms
# wins: 62, losses: 38, draws: 0
# total games: 100
#
#
#
# depth 3, 100ms
# wins: 93, losses: 7, draws: 0
# total games: 100
#
# depth 3, 500ms

#run_beginner_bot_games(3, 2000, 10, save_games=True) # 6 wins, 4 losses

# tactition time 100, and search depth 1, taktition is slightly stronger

print('depth 3, model 6_1, taktition 2000 ms')
run_taktician_bot_games(3, 2000, 100, save_games=True) # 1 win, 9 losses

# run_tiltak_bot_games(1, 100, 10, save_games=False) # 0 wins, 10 losses

#run_tiltak_bot_games(2, 100, 10, save_games=True) # 0 wins, 9 losses, 1 draw

# run_tiltak_bot_games(3, 100, 10, save_games=True)
