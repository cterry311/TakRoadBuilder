from engine_clients.beginner_bot import BeginnerBotEngine
from engine_clients.taktician_bot import TakticianTEI
from engine_clients.tiltak_bot import Tiltak

import torch
from model.tak_net import TakNet

import tak_engine


from engine_clients.roadbuilder import RoadBuilder

model_client = RoadBuilder('model_checkpoints/version_6_1.pt')


def get_model_move(game: tak_engine.Tak, search_time, c_puct, urgency):
    tps = game.to_tps()
    best_move = model_client.best_move(tps, time_limit=search_time, c_puct=c_puct, urgency=urgency)[0]
    return best_move




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


def run_beginner_bot_games(beginner_bot_time, model_time, c_puct, urgency, num_games, save_games=False):
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
                move = get_model_move(game, model_time, c_puct, urgency)
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
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/game_beginnerBot_{i}_mcst.ptn')
    print(f'wins: {wins}, losses: {losses}, draws: {draws}')
    print(f'total games: {wins + losses + draws}')
    return wins, losses, draws


def run_taktician_bot_games(taktician_bot_time, model_time, c_puct, urgency, num_games, save_games=False):
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
                move = get_model_move(game, model_time, c_puct, urgency)
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
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/game_tactician_{i}_mcst_short_opening.ptn')
    print(f'wins: {wins}, losses: {losses}, draws: {draws}')
    print(f'total games: {wins + losses + draws}')
    return wins, losses, draws


def run_tiltak_bot_games(taktician_bot_time, model_time, c_puct, urgency, num_games, save_games=False):
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
                move = get_model_move(game, model_time, c_puct, urgency)
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
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/game_tiltak_{i}_mcst_short_opening.ptn')
    print(f'wins: {wins}, losses: {losses}, draws: {draws}')
    print(f'total games: {wins + losses + draws}')
    return wins, losses, draws


# print('beginner bot games, mutual 200 ms think time')
# run_beginner_bot_games(200, 200, 1.5, -0.2, 100)
# wins: 99, losses: 1, draws: 0
# total games: 100


# print('taktition bot games, mutual 2000 ms')
# run_taktician_bot_games(2000, 2000, 1.5, -0.2, 100, save_games=True)
# wins: 83, losses: 16, draws: 1
# total games: 100


# print('tiltak bot games, mutual 2000 ms')
# run_tiltak_bot_games(2000, 2000, 1.5, -0.2, 100, save_games=True)
# wins: 2, losses: 98, draws: 0
# total games: 100




# print('tiltak bot games, mutual 2000 ms, low depth opening')
# run_tiltak_bot_games(2000, 2000, 1.5, -0.2, 100, save_games=True)
# wins: 23, losses: 73, draws: 4
# total games: 100



# print('taktition bot games, mutual 2000 ms, low depth opening')
# run_taktician_bot_games(2000, 2000, 1.5, -0.2, 100, save_games=True)
# wins: 84, losses: 16, draws: 0
# total games: 100



# print('taktition bot games, mutual 2000 ms')
# run_taktician_bot_games(2000, 2000, 1.5, 0.0, 100, save_games=True)
# wins: 95, losses: 5, draws: 0
# total games: 100




# print('tiltak bot games, mutual 2000 ms, low depth opening')
# run_tiltak_bot_games(2000, 2000, 1.5, 0.0, 100, save_games=True)
# wins: 18, losses: 81, draws: 1
# total games: 100



