"""
play games between the model with search tree and a tei engine, or between two tei engines
"""
from engine_clients.roadbuilder import RoadBuilder, RoadBuilderGame
from engine_clients.teiInterface import TeiEngine
import tak_engine



def save_moves_to_ptn(game: tak_engine.Tak, moves: list[str], player_white: str, player_black: str, save_path: str):
    """
    saves a sequence of moves as a valid ptn file

    :param game: a tak_engine.Tak instance, the game of the file to save
    :param moves: a list of strings, each element in the list is a move taken
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


def roadbuilder_vs_tei(num_games: int, model_path: str, exe: str, time_limit: int=None, node_limit: int=None, depth_limit: int=None, save_games: bool=True):
    """
    plays the designated number of games between the model at the given path and the provided engine tei.
    will log results to the console

    :param num_games: the number of games to play
    :param model_path: the path to the model to use
    :param exe: the exe to use found in engine_clients/exe
    :param time_limit: the time limit in milliseconds per move
    :param node_limit: sim limit per move
    :param depth_limit: avg depth limit per move
    :param save_games: if the games are saved or not
    :return:
    """
    tei_engine = TeiEngine(exe=exe, half_komi=4)

    wins = 0
    losses = 0
    draws = 0

    white_wins = 0
    black_wins = 0

    white_draws = 0
    black_draws = 0

    white_losses = 0
    black_losses = 0

    for i in range(89, num_games):
        game = tak_engine.Tak(komi=2)
        r_builder = RoadBuilderGame(model_path)
        moves = []
        k = i
        while True:
            if k % 2 == 0:
                move, _, _, _ = r_builder.best_move(max_sims=node_limit, time_limit=time_limit, depth_limit=depth_limit)
                moves.append(move)

                game.make_move(move)
                r_builder.make_move(move)

                if game.result() != 0:
                    break
            else:
                move = tei_engine.best_move(moves=moves, movetime_ms=time_limit, nodes=node_limit, depth=depth_limit)
                moves.append(move)

                game.make_move(move)
                r_builder.make_move(move)

                if game.result() != 0:
                    break
            k += 1
        game_result = game.result()
        if game_result == 3:
            draws += 1
            if i % 2 == 0:
                white_draws += 1
            else:
                black_draws += 1
        elif i % 2 == 0:
            if game_result == 1:
                wins += 1
                white_wins += 1
            elif game_result == 2:
                losses += 1
                white_losses += 1
            else:
                raise ValueError(f'game result {game_result} not recognized')
        elif i % 2 == 1:
            if game_result == 2:
                wins += 1
                black_wins += 1
            elif game_result == 1:
                losses += 1
                black_losses += 1
            else:
                raise ValueError(f'game result {game_result} not recognized')
        else:
            raise ValueError(f'game result {game_result} not recognized')
        if save_games:
            model_name = model_path.split('/')[-1].split('.')[0]
            engine_name = exe.split('/')[-1].split('.')[0]

            player_white = model_name if i % 2 == 0 else engine_name
            player_black = model_name if i % 2 == 1 else engine_name
            save_moves_to_ptn(game, moves, player_white, player_black, f'ptn_files/{model_name}_vs_{engine_name}_{i}.ptn')
    model_name = model_path.split('/')[-1].split('.')[0]
    engine_name = exe.split('/')[-1].split('.')[0]
    print(f"{model_name} vs {engine_name}")
    print(f'overview: wins: {wins}, draws: {draws}, losses: {losses}, expected_value: {(wins - losses) / num_games}')
    print(f'white: wins: {white_wins}, draws: {white_draws}, losses: {white_losses}, expected_value: {(white_wins - white_losses) / (white_wins + white_draws + white_losses)}')
    print(f'black: wins: {black_wins}, draws: {black_draws}, losses: {black_losses}, expected_value: {(black_wins - black_losses) / (black_wins + black_draws + black_losses)}')



def tei_vs_tei(num_games: int, exe1: str, exe2: str, time_limit: int=None, node_limit: int=None, depth_limit: int=None, save_games: bool=True):
    """
    plays games between two tei engines, exe1 and exe2, will log results to the console

    :param num_games: the number of games to play
    :param exe1: the first exe to use, found in engine_clients/exe
    :param exe2: the second exe to use, found in engine_clients/exe
    :param time_limit: the time limit in milliseconds per move
    :param node_limit: the sim limit per move
    :param depth_limit: the avg depth limit per move
    :param save_games: if to save the games or not
    :return:
    """
    tei_engine_1 = TeiEngine(exe=exe1, half_komi=4)
    tei_engine_2 = TeiEngine(exe=exe2, half_komi=4)

    wins = 0
    losses = 0
    draws = 0

    white_wins = 0
    black_wins = 0

    white_draws = 0
    black_draws = 0

    white_losses = 0
    black_losses = 0

    for i in range(num_games):
        game = tak_engine.Tak(komi=2)
        moves = []
        k = i
        while True:
            if k % 2 == 0:
                move = tei_engine_1.best_move(moves=moves, movetime_ms=time_limit, nodes=node_limit, depth=depth_limit)
                moves.append(move)

                game.make_move(move)

                if game.result() != 0:
                    break
            else:
                move = tei_engine_2.best_move(moves=moves, movetime_ms=time_limit, nodes=node_limit, depth=depth_limit)
                moves.append(move)

                game.make_move(move)

                if game.result() != 0:
                    break
            k += 1
        game_result = game.result()
        if game_result == 3:
            draws += 1
            if i % 2 == 0:
                white_draws += 1
            else:
                black_draws += 1
        elif i % 2 == 0:
            if game_result == 1:
                wins += 1
                white_wins += 1
            elif game_result == 2:
                losses += 1
                white_losses += 1
            else:
                raise ValueError(f'game result {game_result} not recognized')
        elif i % 2 == 1:
            if game_result == 2:
                wins += 1
                black_wins += 1
            elif game_result == 1:
                losses += 1
                black_losses += 1
            else:
                raise ValueError(f'game result {game_result} not recognized')
        else:
            raise ValueError(f'game result {game_result} not recognized')
        if save_games:
            engine_name_1 = exe1.split('/')[-1].split('.')[0]
            engine_name_2 = exe2.split('/')[-1].split('.')[0]

            player_white = engine_name_1 if i % 2 == 0 else engine_name_2
            player_black = engine_name_1 if i % 2 == 1 else engine_name_2
            save_moves_to_ptn(game, moves, player_white, player_black,
                              f'ptn_files/{engine_name_1}__vs__{engine_name_2}__{i}.ptn')
    engine_name_1 = exe1.split('/')[-1].split('.')[0]
    engine_name_2 = exe2.split('/')[-1].split('.')[0]
    print(f"{engine_name_1} vs {engine_name_2}")
    print(f'overview: wins: {wins}, draws: {draws}, losses: {losses}, expected_value: {(wins - losses) / num_games}')
    print(
        f'white: wins: {white_wins}, draws: {white_draws}, losses: {white_losses}, expected_value: {(white_wins - white_losses) / (white_wins + white_draws + white_losses)}')
    print(
        f'black: wins: {black_wins}, draws: {black_draws}, losses: {black_losses}, expected_value: {(black_wins - black_losses) / (black_wins + black_draws + black_losses)}')


# print('games against roadbuilder')
# roadbuilder_vs_tei(200, 'model_checkpoints/version_13_60.pt', 'taktician.exe', save_games=True, time_limit=2000)
# print()
# roadbuilder_vs_tei(200, 'model_checkpoints/version_13_60.pt', 'tiltak.exe', save_games=True, time_limit=2000)
# print()
# print('finishing benchmarking on roadbuilder vs topaz')
# roadbuilder_vs_tei(200, 'model_checkpoints/version_13_60.pt', 'topaz.exe', save_games=True, time_limit=2000)
# print()
# roadbuilder_vs_tei(200, 'model_checkpoints/version_13_60.pt', 'takson.exe', save_games=True, time_limit=2000)
# print()
# roadbuilder_vs_tei(200, 'model_checkpoints/version_13_60.pt', 'syntaks.exe', save_games=True, time_limit=2000)
# print()
# roadbuilder_vs_tei(200, 'model_checkpoints/version_13_60.pt', 'takzero.exe', save_games=True, time_limit=2000)
# print()
# print('tei against tei')
# print('taktition against tei')
# tei_vs_tei(100, 'taktician.exe', 'tiltak.exe', save_games=True, time_limit=2000)
# print()
# print('tiltak against tei')
# tei_vs_tei(100, 'tiltak.exe', 'topaz.exe', save_games=True, time_limit=2000)
# print()
#
# print('topaz against tei')
# tei_vs_tei(100, 'topaz.exe', 'takson.exe', save_games=True, time_limit=2000)
# print()
# tei_vs_tei(100, 'topaz.exe', 'syntaks.exe', save_games=True, time_limit=2000)
# print()
# print('takson against tei')
# tei_vs_tei(100, 'takson.exe', 'syntaks.exe', save_games=True, time_limit=2000)
# print()
#
# print('takzero vs tei')
# tei_vs_tei(100, 'takzero.exe', 'tiltak.exe', save_games=True, time_limit=2000)
# print()
# tei_vs_tei(100, 'takzero.exe', 'topaz.exe', save_games=True, time_limit=2000)
# print()
# tei_vs_tei(100, 'takzero.exe', 'takson.exe', save_games=True, time_limit=2000)
# print()
# tei_vs_tei(100, 'takzero.exe', 'syntaks.exe', save_games=True, time_limit=2000)
# print()
#
#
# tei_vs_tei(100, 'tiltak.exe', 'takson.exe', save_games=True, time_limit=2000)
# print()
#
# tei_vs_tei(100, 'takzero.exe', 'taktician.exe', save_games=True, time_limit=2000)
# print()
# tei_vs_tei(100, 'taktician.exe', 'topaz.exe', save_games=True, time_limit=2000)
# print()
# tei_vs_tei(100, 'taktician.exe', 'takson.exe', save_games=True, time_limit=2000)
# print()
# tei_vs_tei(100, 'taktician.exe', 'syntaks.exe', save_games=True, time_limit=2000)
# print()
#
#
# tei_vs_tei(100, 'tiltak.exe', 'syntaks.exe', save_games=True, time_limit=2000)
# print()



roadbuilder_vs_tei(100, 'model_checkpoints/version_14_62.pt', 'takzero.exe', save_games=True, time_limit=500)




# games against roadbuilder
# version_13_60 vs taktician
# overview: wins: 199, draws: 0, losses: 1, expected_value: 0.99
# white: wins: 100, draws: 0, losses: 0, expected_value: 1.0
# black: wins: 99, draws: 0, losses: 1, expected_value: 0.98
# +920 elo
# interval [+727, inf) because of the massive gap in strength, there is a substantial limit on how accurate I can be and how useful these results are
#
# version_13_60 vs tiltak
# overview: wins: 190, draws: 2, losses: 8, expected_value: 0.91
# white: wins: 100, draws: 0, losses: 0, expected_value: 1.0
# black: wins: 90, draws: 2, losses: 8, expected_value: 0.82
# +531 elo
# interval [438, 709]
#
# version_13_60 vs takzero
# ins: 19, losses: 165, draws: 16, expected_value: 0.135
# white wins: 8, white losses: 84, white draws: 8, expected_value: 0.12
# black wins: 11, black losses: 81, black draws: 8, expected_value: 0.15
# -323 elo
# range [-408, -261]
#
# version_13_60 vs topaz
# wins: 13, losses: 166, draws: 21, expected_value: -0.765
# white wins: 10, white losses: 75, white draws: 15, expected_value: -0.65
# black wins: 3, black losses: 91, black draws: 6, expected_value: -0.88
# -350 elo
# interval [-442, -285]
#
# version_13_60 vs takson
# overview: wins: 0, draws: 0, losses: 200, expected_value: -1.0
# white: wins: 0, draws: 0, losses: 100, expected_value: -1.0
# black: wins: 0, draws: 0, losses: 100, expected_value: -1.0
# NaN elo
# interval (-inf, -727]
#
# version_13_60 vs syntaks
# overview: wins: 0, draws: 0, losses: 200, expected_value: -1.0
# white: wins: 0, draws: 0, losses: 100, expected_value: -1.0
# black: wins: 0, draws: 0, losses: 100, expected_value: -1.0
# NaN elo
# interval (-inf, -727]
#



# taktician vs tiltak
# overview: wins: 1, draws: 0 losses: 99, expected_value: -0.98
# white: wins: 1, draws: 0, losses: 49, expected_value: -0.96
# black: wins: 0, draws: 0, losses: 50, expected_value: -1.0
# -920 elo
# interval [+727, inf)
#
# tiltak vs topaz
# overview: wins: 0, draws: 0, losses: 100, expected_value: -1.0
# white: wins: 0, draws: 0, losses: 50, expected_value: -1.0
# black: wins: 0, draws: 0, losses: 50, expected_value: -1.0



# 13_36 vs takzero
# wins: 19, losses: 65, draws: 16
# white wins: 4, white losses: 41, white draws: 5
# black wins: 15, black losses: 24, black draws: 11
# -173 elo

# 13_62 vs takzero
# wins: 3, losses: 83, draws: 14
# white wins: 0, white losses: 43, white draws: 7
# black wins: 3, black losses: 40, black draws: 7