import sqlite3

from game_emulator.game import Tak
from random import choice

conn = sqlite3.connect("games.db")
cursor = conn.cursor()


# cursor.execute("PRAGMA table_info(filtered_1);")
# columns = cursor.fetchall()
#
# print(f"Schema for 'your_table_name':")
# for col in columns:
#     cid, name, dtype, notnull, default, pk = col
#     print(f" - {name} ({dtype}){' PRIMARY KEY' if pk else ''}{' NOT NULL' if notnull else ''}")

'''
 - id (INTEGER) PRIMARY KEY
 - date (INT)
 - size (INT)
 - player_white (VARCHAR(20))
 - player_black (VARCHAR(20))
 - notation (TEXT)
 - result (VARCAR(10))
 - timertime (INT)
 - timerinc (INT)
 - rating_white (INT)
 - rating_black (INT)
 - unrated (INT)
 - tournament (INT)
 - komi (INT)
 - pieces (INT)
 - capstones (INT)
 - rating_change_white (INT)
 - rating_change_black (INT)
 - extra_time_amount (INT)
 - extra_time_trigger (INT)
'''

# cursor.execute('''
#     SELECT COUNT(*) FROM games
# ''')
# print(cursor.fetchall())  # 863,151
#
# cursor.execute('''
#     SELECT COUNT(*) FROM games
#     WHERE size = 6 AND komi = 0 AND pieces = 30 AND capstones = 1
# ''') # 83,842
# print(cursor.fetchall())
#
#
# List all tables
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';") # games is the only table
# print(cursor.fetchall())
#
# Look at a table's contents
# cursor.execute("SELECT * FROM your_table_name LIMIT 10;")
# for row in cursor.fetchall():
#     print(row)

def parse_server_notation(notation : str, result):
    split_moves = notation.split(',')
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
    return ptn_notation

# cursor.execute('''
#     SELECT notation from filtered_3
#
# ''') #
# output = cursor.fetchall()
# print(output[1000][0])
# print('\n\n\n')
# print(parse_server_notation(output[1000][0], '1-0'))
# conn.close()

# cursor.execute('''
#     SELECT * FROM games_formated
#     WHERE id = 418665
# ''')
#
# output = cursor.fetchall()
#
# print('id: ', output[0][0])
# print('player_white: ', output[0][1])
# print('player_black: ', output[0][2])
# print('notation: ', output[0][3])
# print('game_length: ', output[0][4])
#
# with open('wronged.ptn', 'w') as f:
#     f.write(output[0][3])




#
# cursor.execute('''
#     SELECT * FROM training_positions WHERE id = 2640
# ''')
#
# position_id, game_id, tps, move, move_number, edited_position = cursor.fetchall()[0]
# tak = Tak.from_tps(tps)
# legal_moves = tak.get_all_legal_moves()
# filtered_moevs, result = tak.get_all_legal_moves(filtering=1, do_pruning=False, include_result=True)
# # print('legal moves ', legal_moves)
# # print('filtered moves ', filtered_moevs)
# print('result ', result)
# print(move in legal_moves)
# print(move in filtered_moevs)
# print(f'tps: {tps}')
#
# for move_i in legal_moves:
#     if move_i not in filtered_moevs:
#         print(f"legal move {move_i} not in filtered moves")
#
# for move_i in filtered_moevs:
#     if move_i not in legal_moves:
#         print(f"filtered move {move_i} not in legal moves")
#
#
# cursor.execute('''
#     SELECT * FROM training_positions WHERE id = 349741
# ''')
#
# position_id, game_id, tps, move, move_number, edited_position, cleaned_position = cursor.fetchall()[0]
# tak = Tak.from_tps(tps)
# possible_moves = tak.get_all_legal_moves()
# for possible_move in possible_moves:
#     tak.make_move(possible_move)
#     new_possible_moves = tak.get_all_legal_moves()
#     for new_possible_move in new_possible_moves:
#         try:
#             tak.make_move(new_possible_move)
#             tak.undo_move()
#         except Exception as e:
#             print(possible_move, new_possible_move)
#             raise e
#     tak.undo_move()
# print('went through all moves')
# filtered_moves, result = tak.get_all_legal_moves(filtering=1, do_pruning=False, include_result=True)
# print(move in possible_moves)
# print(move in filtered_moves)
# cursor.close()



print('filtering training positions...')
cursor.execute('''
    SELECT * FROM training_positions WHERE cleaned_move = 0
''')
positions = cursor.fetchall()
num_positions = len(positions)
print(f'positions to filter: {num_positions}')
print(('-' * 46) + 'progress' + ('-' * 46))
interval = num_positions // 100
i = 0
for position in positions:
    position_id, game_id, tps, move, move_number, edited_position, cleaned_move = position
    try:
        tak = Tak.from_tps(tps)
        filtered_moves, result = tak.get_all_legal_moves(filtering=1, do_pruning=False, include_result=True)
    except Exception as e:
        print(f'\nposition id: {position_id}')
        raise e
    if move not in filtered_moves:
        # the move made is not within the filtered moves
        if result == 1:
            # there is a winning move avaliable, so I will edit the position such that they make the winning move instead
            rand_winning_move = choice(filtered_moves)
            cursor.execute('''
                UPDATE training_positions 
                SET edited_position = 1, move = ?, cleaned_move = 1
                WHERE id = ?
            ''', (rand_winning_move, position_id))
        elif result == 0:
            # there is no winning move availible and they played a losing move when a neutral move was availible, drop the position
            cursor.execute('''
                DELETE FROM training_positions WHERE id = ?
            ''', (position_id,))
        elif result == -1:
            legal_moves = tak.get_all_legal_moves()
            if move in legal_moves:
                cursor.execute('''
                    DELETE FROM training_positions WHERE id = ?
                ''', (position_id,))
            else:
                raise ValueError(f'move {move} is not inside the set of legal moves for position {position_id}')
        else:
            # somehow result is not 0, 1 or -1, this branch should never be reached
            raise ValueError(f'result {result} is not 0, 1 or -1 for position {position_id}')
    else:
        cursor.execute('''
            UPDATE training_positions 
            SET cleaned_move = 1
            WHERE id = ?
        ''', (position_id,))
    i += 1
    if i % interval == 0:
        print('*', end='', flush=True)
        conn.commit()
conn.commit()
print('\ntraining positions filtered\n')
print('done')