import sqlite3
import re
from game_emulator.game import Tak

def parse_server_notation(notation : str, result):
    split_moves = notation.split(',')
    if len(split_moves) == 1:
        if split_moves[0] == '':
            split_moves = []
    ptn_notation = f'[Result "{result}"]\n[Size "6"]\n[Flats "30"]\n[Caps "1"]\n'
    i = 0
    include_number = True
    for move in split_moves:
        if include_number:
            i += 1
            ptn_notation += f'\n{i}.'
            include_number = False
        else:
            include_number = True
        move_elements = move.split(' ')
        if move_elements[0] == 'P':
            ptn_move = ''
            if len(move_elements) == 3:
                if move_elements[2] == 'C':
                    ptn_move += 'C'
                elif move_elements[2] == 'W':
                    ptn_move += 'S'
                else:
                    raise ValueError(f'Invalid move format {move}')
            ptn_move += move_elements[1].lower()
            ptn_notation += f' {ptn_move}'
        elif move_elements[0] == 'M':
            origin = move_elements[1].lower()
            destination = move_elements[2].lower()
            drop_pattern = list(map(lambda x: int(x), move_elements[3:]))
            ptn_move = ''
            ptn_move += str(sum(drop_pattern))
            ptn_move += origin
            direction = ''
            if origin[0] > destination[0]:
                direction = '<'
            elif origin[0] < destination[0]:
                direction = '>'
            elif int(origin[1]) > int(destination[1]):
                direction = '-'
            elif int(origin[1]) < int(destination[1]):
                direction = '+'
            else:
                raise ValueError(f'Invalid move format {move}')
            ptn_move += direction
            for num in drop_pattern:
                ptn_move += str(num)
            ptn_notation += f' {ptn_move}'
        else:
            raise ValueError(f'Invalid move format {move}')
    ptn_notation += f'\n{result}'
    return ptn_notation, len(split_moves)


def get_moves_ordered(notation : str) -> list[str]:
    rows = notation.split('\n')
    moves = []
    for row in rows:
        if re.match(r'\d+\. ', row):
            parts = row.split(' ')
            moves.append(parts[1])
            if len(parts) == 3:
                moves.append(parts[2])
    return moves


conn = sqlite3.connect("games.db")
cursor = conn.cursor()

cursor.execute('''
    DROP TABLE IF EXISTS filtered_1;
''')
cursor.execute('''
    DROP TABLE IF EXISTS filtered_2;
''')

cursor.execute('''
    DROP TABLE IF EXISTS filtered_3;
''')

cursor.execute('''
    DROP TABLE IF EXISTS games_formated;
''')

cursor.execute('''
    CREATE TABLE filtered_1 AS
    SELECT id, player_white, player_black, notation, result, timertime, timerinc, rating_white, rating_black, unrated, tournament, rating_change_white, rating_change_black, extra_time_amount, extra_time_trigger FROM games
    WHERE size = 6 AND komi = 0 AND pieces = 30 AND capstones = 1
''') # 83,842

cursor.execute('''
    CREATE TABLE filtered_2 AS
    SELECT id, player_white, player_black, notation, result, timertime, timerinc, extra_time_amount, extra_time_trigger from filtered_1
    WHERE rating_white > 1500 AND rating_black > 1500 AND result != '0-1' AND result != '1-0' AND result != '0-0'
''') # 16279\

cursor.execute('''
    CREATE TABLE filtered_3 AS
    SELECT id, player_white, player_black, notation, result from filtered_2
    WHERE timertime > 600 AND timerinc > 10
''')

cursor.execute('''
    CREATE TABLE games_formated (
    id INTEGER PRIMARY KEY,
    player_white VARCHAR(20),
    player_black VARCHAR(20),
    notation TEXT,
    game_length INT
);
''')


cursor.execute('''
    CREATE TABLE training_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER,
        tps TEXT,
        move VARCHAR(9),
        move_number SMALLINT,
        FOREIGN KEY (game_id) REFERENCES games_formated(id)
    )
''')

cursor.execute('''
    SELECT id, player_white, player_black, notation, result FROM filtered_3
''')

all_games = cursor.fetchall()

for game in all_games:
    game_id, player_white, player_black, notation, result = game
    try:
        ptn_notation, game_length = parse_server_notation(notation, result)
    except Exception as e:
        print(f"game id: {game_id}")
        print(f"player white: {player_white}")
        print(f"player black: {player_black}")
        print(f"notation: {notation}")
        print(f"result: {result}")
        raise e
    cursor.execute('''
        INSERT INTO games_formated (id, player_white, player_black, notation, game_length)
            VALUES (?, ?, ?, ?, ?)
    ''', (game_id, player_white, player_black, ptn_notation, game_length))
conn.commit()

cursor.execute('''
    SELECT id, player_white, player_black, notation, game_length FROM games_formated
''')

games_formated = cursor.fetchall()
for game in games_formated:
    game_id, player_white, player_black, notation, game_length = game
    moves = get_moves_ordered(notation)
    tak = Tak(6)
    for move in moves:
        tak.make_move(move)
    if tak.turn not in [4, 5, 6]:
        cursor.execute('''
            DELETE FROM games_formated WHERE id = ?
        ''', (game_id,))
        conn.commit()

cursor.execute('''
    SELECT id, player_white, player_black, notation, game_length FROM games_formated
''')

games_formated = cursor.fetchall()
for game in games_formated:
    game_id, player_white, player_black, notation, game_length = game
    moves = get_moves_ordered(notation)
    tak = Tak(6)
    move_number = 0
    for move in moves:
        move_number += 1
        tps = tak.to_tps()
        cursor.execute('''
            INSERT INTO training_positions (game_id, tps, move, move_number)
                VALUES (?, ?, ?, ?)
        ''', (game_id, tps, move, move_number))
        tak.make_move(move)
conn.commit()


positions = cursor.execute('''
    SELECT * FROM training_positions
''')
for position in positions:
    position_id, game_id, tps, move, move_number = position
    tak = Tak.from_tps(tps)


conn.close()
