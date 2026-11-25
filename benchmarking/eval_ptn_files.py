import tak_engine
import sys
import os

FILE_START = 'version_14_36_vs_takzero_'


files = os.listdir('ptn_files')

wins = 0
draws = 0
losses = 0

white_wins = 0
white_draws = 0
white_losses = 0

black_wins = 0
black_draws = 0
black_losses = 0

for file in files:
    if file.startswith(FILE_START):
        i = int(file.split('.')[0][len(FILE_START):])
        game = tak_engine.Tak(komi=2)
        with open(f'ptn_files/{file}', 'r') as f:
            content = f.read()
            lines = content.split('\n')
            in_game = False
            for line in lines:
                if line.startswith('1. '):
                    in_game = True
                if not in_game:
                    continue

                parts = line.split()  # no argument: ignores trailing/double spaces
                if len(parts) < 2:
                    continue

                game.make_move(parts[1])  # white
                if len(parts) >= 3:
                    game.make_move(parts[2])  # black
        result = game.result()
        if result == 0:
            raise RuntimeError(f'game is not terminal {file}')
        elif result == 1:
            # white won
            if i % 2 == 0:
                # player 1 is white
                wins += 1
                white_wins += 1
            else:
                # player 1 is black
                losses += 1
                black_losses += 1
        elif result == 2:
            # black won
            if i % 2 == 0:
                # player 1 is white
                losses += 1
                white_losses += 1
            else:
                # player 1 is black
                wins += 1
                black_wins += 1
        elif result == 3:
            # draw
            draws += 1
            if i % 2 == 0:
                # player 1 is white
                white_draws += 1
            else:
                # player 1 is black
                black_draws += 1
        elif result == 4:
            pass
        else:
            raise RuntimeError(f'unknown result {result}')


print(f'wins: {wins}, losses: {losses}, draws: {draws}')
print(f'white wins: {white_wins}, white losses: {white_losses}, white draws: {white_draws}')
print(f'black wins: {black_wins}, black losses: {black_losses}, black draws: {black_draws}')