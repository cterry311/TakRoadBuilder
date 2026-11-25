"""
play games against BeginnerBot, Taktician, and Tiltak.
model uses search tree now
"""
from engine_clients.beginner_bot import BeginnerBotEngine
from engine_clients.taktician_bot import TakticianTEI
from engine_clients.tiltak_bot import Tiltak

import torch
from model.tak_net import TakNet

import tak_engine


from engine_clients.roadbuilder import RoadBuilder

model_client = RoadBuilder('model_checkpoints/version_13_60.pt')


def get_model_move(game: tak_engine.Tak, search_time: int, c_puct: float, urgency: float) -> str:
    """
    get a move from the model, given a game and search time

    :param game: the game to take the move from, the current position is used as the state
    :param search_time: the time in milliseconds to search for a move
    :param c_puct: how heavily the initial policy is weighted
    :param urgency: how much to prioritize searching unexplored nodes
    :return: the move to make in ptn format
    """
    tps = game.to_tps()
    best_move = model_client.best_move(tps, time_limit=search_time, c_puct=c_puct, urgency=urgency, komi=2)[0]
    return best_move




def save_moves_to_ptn(game: tak_engine.Tak, moves: list[str], player_white: str, player_black: str, save_path: str):
    """
    saves a move sequence to a ptn file

    :param game: the game to save the moves from
    :param moves: the move sequence as ptn moves
    :param player_white: the white player
    :param player_black: the black player
    :param save_path: the path to save the ptn file to
    :return:
    """
    ptn_game = f'[size "6"]\n[player1 "{player_white}"]\n[player2 "{player_black}"]\n[komi "2"]\n'

    for i, move in enumerate(moves):
        if i % 2 == 0:
            ptn_game += f'{(i // 2 ) + 1}. '
        ptn_game += move + ' '
        if i % 2 == 1:
            ptn_game += '\n'

    with open(save_path, 'w') as f:
        f.write(ptn_game)


def run_beginner_bot_games(beginner_bot_time: int, model_time: int, c_puct: float, urgency: float, num_games: int, save_games: bool=False):
    """
    run games against beginnerBot using the search tree, will log results to the console

    :param beginner_bot_time: the time in milliseconds allotted for beginnerbot
    :param model_time: the time in milliseconds allotted for the model to search
    :param c_puct: how heavily the initial policy is weighted
    :param urgency: how much to prioritize searching unexplored nodes
    :param num_games: how many games to play
    :param save_games: if to save the games
    :return:
    """
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


def run_taktician_bot_games(taktician_bot_time: int, model_time: int, c_puct: float, urgency: float, num_games: int, save_games: bool=False):
    """
    play games against taktician, will log results to the console

    :param taktician_bot_time: the time in milliseconds allotted for taktician
    :param model_time: the time in milliseconds allotted for the model to search
    :param c_puct: how heavily the initial policy is weighted
    :param urgency: how much to prioritize searching unexplored nodes
    :param num_games: the number of games to play
    :param save_games: if to save the games
    :return:
    """
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


def run_tiltak_bot_games(taktician_bot_time: int, model_time: int, c_puct: float, urgency: float, num_games: int, save_games: bool=False):
    """
    play games against tiltak, will log results to the console

    :param taktician_bot_time: the time in milliseconds allotted for taktician
    :param model_time: the time in milliseconds allotted for the model to search
    :param c_puct: how heavily the initial policy is weighted
    :param urgency: how much to prioritize searching unexplored nodes
    :param num_games: how many games to play
    :param save_games: if to save the games
    :return:
    """
    bot = Tiltak(half_komi=4)
    wins = 0
    draws = 0
    losses = 0
    for i in range(num_games):
        game = tak_engine.Tak(komi=2)
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
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/game_tiltak_{i}_9_5_long.ptn')
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


# print('tiltak bot games, mutual 2000 ms') # version_9_5
# run_tiltak_bot_games(2000, 2000, 1.5, -0.2, 100, save_games=True)
# wins: 82, losses: 17, draws: 1
# total games: 100


# print('tiltak bot games, mutual 500 ms') # version_9_34.pt
# run_tiltak_bot_games(500, 500, 1.5, -0.2, 100, save_games=True)
# wins: 81, losses: 13, draws: 6
# total games: 100


# print('tiltak bot games, mutual 2000 ms') # version_9_34.pt
# run_tiltak_bot_games(2000, 2000, 1.5, -0.2, 100, save_games=True)
# wins: 86, losses: 13, draws: 1
# total games: 100



# print('tiltak bot games, mutual 2000 ms') # version_13_60.pt
# run_tiltak_bot_games(2000, 2000, 1.5, -0.2, 100, save_games=True)
# wins: 82, losses: 13, draws: 5
# total games: 100
