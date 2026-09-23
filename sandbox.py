import sqlite3

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





cursor.execute('''
    SELECT * FROM training_positions
''')

position = cursor.fetchall()[11000]

print(position)

cursor.close()