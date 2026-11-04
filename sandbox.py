# import sqlite3
import tak_engine

from python_game.game import Tak
from random import choice

# conn = sqlite3.connect("games.db")
# cursor = conn.cursor()
#
#
# # cursor.execute("PRAGMA table_info(filtered_1);")
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

# def parse_server_notation(notation : str, result):
#     split_moves = notation.split(',')
#     ptn_notation = f'[Result "{result}"]\n[Size "6"]\n[Flats "30"]\n[Caps "1"]\n'
#     i = 0
#     include_number = True
#     for move in split_moves:
#         if include_number:
#             i += 1
#             ptn_notation += f'\n{i}.'
#             include_number = False
#         else:
#             include_number = True
#         move_elements = move.split(' ')
#         if move_elements[0] == 'P':
#             ptn_move = ''
#             if len(move_elements) == 3:
#                 if move_elements[2] == 'C':
#                     ptn_move += 'C'
#                 elif move_elements[2] == 'W':
#                     ptn_move += 'S'
#                 else:
#                     raise ValueError(f'Invalid move format {move}')
#             ptn_move += move_elements[1].lower()
#             ptn_notation += f' {ptn_move}'
#         elif move_elements[0] == 'M':
#             origin = move_elements[1].lower()
#             destination = move_elements[2].lower()
#             drop_pattern = list(map(lambda x: int(x), move_elements[3:]))
#             ptn_move = ''
#             ptn_move += str(sum(drop_pattern))
#             ptn_move += origin
#             direction = ''
#             if origin[0] > destination[0]:
#                 direction = '<'
#             elif origin[0] < destination[0]:
#                 direction = '>'
#             elif int(origin[1]) > int(destination[1]):
#                 direction = '-'
#             elif int(origin[1]) < int(destination[1]):
#                 direction = '+'
#             else:
#                 raise ValueError(f'Invalid move format {move}')
#             ptn_move += direction
#             for num in drop_pattern:
#                 ptn_move += str(num)
#             ptn_notation += f' {ptn_move}'
#         else:
#             raise ValueError(f'Invalid move format {move}')
#     ptn_notation += f'\n{result}'
#     return ptn_notation

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



# print('filtering training positions...')
# cursor.execute('''
#     SELECT * FROM training_positions WHERE cleaned_move = 0
# ''')
# positions = cursor.fetchall()
# num_positions = len(positions)
# print(f'positions to filter: {num_positions}')
# print(('-' * 46) + 'progress' + ('-' * 46))
# interval = num_positions // 100
# i = 0
# for position in positions:
#     position_id, game_id, tps, move, move_number, edited_position, cleaned_move = position
#     try:
#         tak = Tak.from_tps(tps)
#         filtered_moves, result = tak.get_all_legal_moves(filtering=1, do_pruning=False, include_result=True)
#     except Exception as e:
#         print(f'\nposition id: {position_id}')
#         raise e
#     if move not in filtered_moves:
#         # the move made is not within the filtered moves
#         if result == 1:
#             # there is a winning move avaliable, so I will edit the position such that they make the winning move instead
#             rand_winning_move = choice(filtered_moves)
#             cursor.execute('''
#                 UPDATE training_positions
#                 SET edited_position = 1, move = ?, cleaned_move = 1
#                 WHERE id = ?
#             ''', (rand_winning_move, position_id))
#         elif result == 0:
#             # there is no winning move availible and they played a losing move when a neutral move was availible, drop the position
#             cursor.execute('''
#                 DELETE FROM training_positions WHERE id = ?
#             ''', (position_id,))
#         elif result == -1:
#             legal_moves = tak.get_all_legal_moves()
#             if move in legal_moves:
#                 cursor.execute('''
#                     DELETE FROM training_positions WHERE id = ?
#                 ''', (position_id,))
#             else:
#                 raise ValueError(f'move {move} is not inside the set of legal moves for position {position_id}')
#         else:
#             # somehow result is not 0, 1 or -1, this branch should never be reached
#             raise ValueError(f'result {result} is not 0, 1 or -1 for position {position_id}')
#     else:
#         cursor.execute('''
#             UPDATE training_positions
#             SET cleaned_move = 1
#             WHERE id = ?
#         ''', (position_id,))
#     i += 1
#     if i % interval == 0:
#         print('*', end='', flush=True)
#         conn.commit()
# conn.commit()


# cursor.execute('''
#     SELECT COUNT(*) FROM training_positions WHERE edited_position = 1
# ''')
# print(cursor.fetchone())
# cursor.close()
#print(0.1+0.2)
#
# import tak_engine
# from model.tak_net import TakNet
# import torch
#
#
#
# def get_model_move(tak, model):
#     model.eval()
#     with torch.no_grad():
#         tensor_rep = torch.from_numpy(tak.to_tensor())
#         legal_mask = torch.from_numpy(tak.get_legal_mask())
#         tensor_rep = tensor_rep.reshape(1, 25, 6, 6)
#         policy, value = model(tensor_rep)
#         value = value.flatten()
#         masked_output = policy.flatten().masked_fill(~(legal_mask.flatten().bool()), -1e9)
#         index_props = torch.softmax(masked_output, dim=0).pow(10)
#         selected_index = torch.multinomial(index_props, 1).squeeze()
#         ptn_move = tak_engine.index_to_ptn(selected_index.item())
#     return ptn_move, value
# model = TakNet()
# model.load_state_dict(torch.load('model_checkpoints/version_1_5.pt')['weights'])
#
# tak = tak_engine.Tak()
# # tak = tak_engine.Tak.from_tps("1,2,1,1,2,1/1,2,211C,1,2,2/2,2,2,x,1,2/1,12S,x,2121S,x,12/1,2,2,12,x2/2,1,1,11112,x,1 2 30")
# # print(get_model_move(tak, model))
#
# ptn_output = '[size "6"]\n\n'
# i = 1
# while True:
#     ptn_output += f'{i}. '
#     ptn_move, values = get_model_move(tak, model)
#     print(ptn_move, values)
#     tak.make_move(ptn_move)
#     ptn_output += ptn_move
#     if tak.result() != 0:
#         break
#     # own_move = input('>')
#     # tak.make_move(own_move)
#     # if tak.result() != 0:
#     #     break
#     ptn_move, values = get_model_move(tak, model)
#     print(ptn_move, values)
#     tak.make_move(ptn_move)
#     ptn_output += ' ' + ptn_move + '\n'
#     if tak.result() != 0:
#         break
#     # own_move =
#     i += 1
#
# print('made it')
# with open('test.ptn', 'w') as f:
#     f.write(ptn_output)
# cursor.close()
# conn.close()


#
# import numpy as np
# import torch
# import tak_engine
# from model.tak_net import TakNet
# import time
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# info = torch.load("model_checkpoints/version_6_1.pt", map_location=device)
# model = TakNet(**info["params"])
# model.load_state_dict(info["weights"])
# model.to(device).eval()                      # eval() matters: BatchNorm must not use batch stats
#
# BATCH = 256
# buf = np.zeros((BATCH, 25, 6, 6), dtype=np.float32)
# flat = buf.reshape(-1)                       # a view onto buf; Rust writes through it
#
# # warm up CUDA so the first real move doesn't pay kernel setup time
# with torch.no_grad():
#     model(torch.zeros(BATCH, 25, 6, 6, device=device))
#
# @torch.no_grad()
# def run_search(tps, max_sims=None, time_limit=None, c_puct=1.5, urgency=-0.2):
#     s = tak_engine.Search(tps, urgency, c_puct)
#     if s.is_over():
#         return None
#
#     start = time.perf_counter()
#     while True:
#         if max_sims is not None and s.sims() >= max_sims:
#             break
#         if time_limit is not None and time.perf_counter() - start >= time_limit:
#             break
#
#         n = s.collect(flat, BATCH)
#         if n == 0:                           # every descent hit a finished game; already backed up
#             continue
#
#         x = torch.from_numpy(buf[:n]).to(device)
#         policy, value = model(x)
#
#         pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
#         vp = torch.softmax(value, dim=1)
#         val = (vp[:, 2] - vp[:, 0]).cpu().numpy()
#
#         s.submit(pol, val)                   # row i here must be row i of buf
#
#     return s.best_move_raw(), s.root_stats(), s.sims()
#
# tps = "1,x5/x,12,1211112C,x,2,12/12,x2,21C,2,112/x,1,1,1,x,2/x3,1,1,2/x3,1,x,2 1 24"
# game = tak_engine.Tak.from_tps(tps)
#
# start_time = time.perf_counter()
# print('started search')
# out = run_search(tps, time_limit=20.0)
# print('finished search')
# if out is not None:
#     raw, stats, sims = out
#     game.make_move_raw(raw)
#     print(tak_engine.Tak.raw_to_ptn(raw), "after", sims, "sims")
#
# print(time.perf_counter() - start_time)

# from engine_clients.roadbuilder import RoadBuilder
# import tak_engine
#
# engine = RoadBuilder('model_checkpoints/version_6_1.pt')
#
# game = tak_engine.Tak()
#
# ptn_game = f'[size "6"]\n[player1 "Player 1"]\n[player2 "Player 2"]\n'
#
# i = 1
# while True:
#     ptn_game += f'{i}. '
#     tps = game.to_tps()
#     best_move = engine.best_move(tps, max_sims=10_000, eps=0)[0]
#     game.make_move(best_move)
#     ptn_game += f'{best_move} '
#     print(best_move)
#     if game.result() != 0:
#         break
#     tps = game.to_tps()
#     best_move = engine.best_move(tps, max_sims=10_000, eps=0)[0]
#     game.make_move(best_move)
#     ptn_game += f'{best_move}\n'
#     print(best_move)
#     if game.result() != 0:
#         break
#     i += 1
#
#
# with open("ptn_files/test2.ptn", "w") as f:
#     f.write(ptn_game)

# from engine_clients.roadbuilder import RoadBuilder
# from model.tak_net import TakNet
# import torch
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# info = torch.load("model_checkpoints/version_6_1.pt", map_location=device)
# model = TakNet(**info["params"])
# model.load_state_dict(info["weights"])
# model.to(device).eval()
#
# engine = RoadBuilder('model_checkpoints/version_6_1.pt')
#
# game = tak_engine.Tak()
# # game.make_move('a1')
# # game.make_move('f6')
# # game.make_move('d4')
#
# print('starting search...')
# move, info, num_simulations, root_value = engine.best_move(game.to_tps(), max_sims=10_000)
# print(f'best move: {move}, num simulations: {num_simulations}, root_value: {root_value}')
# info = sorted(info, key=lambda x: x[1], reverse=True)
# for part in info:
#     print(f"move: {tak_engine.Tak.raw_to_ptn(part[0])}, num visits: {part[1]}, q value: {part[2]},")








import sys
import sqlite3
import tak_engine

# ---- easy-to-swap settings ----
DB_PATH = "self_play_records/self_play_1.db"
GAME_ID = 9                      # overridden by the first command-line argument
OUT_PATH = f"ptn_files/test_{GAME_ID}.ptn"                  # None -> "game_{id}.ptn"

RESULT_TAGS = {1: "1-0", 2: "0-1", 3: "1/2-1/2", 0: "0-0"}   # 0 = hit the ply cap, no result


def export_game(db_path, game_id, out_path=None):
    conn = sqlite3.connect(db_path)
    g = conn.execute(
        """SELECT seed, model, komi, max_sims, temp_plies, n_plies, outcome, end_reason
           FROM self_play_games WHERE id = ?""", (game_id,)).fetchone()
    if g is None:
        raise SystemExit(f"no game with id {game_id} in {db_path}")
    seed, model, komi, max_sims, temp_plies, n_plies, outcome, end_reason = g

    rows = conn.execute(
        "SELECT move_raw FROM self_play_positions WHERE game_id = ? ORDER BY ply",
        (game_id,)).fetchall()
    conn.close()
    raws = [r[0] for r in rows]
    ptn_moves = [tak_engine.Tak.raw_to_ptn(r) for r in raws]

    # sanity check: replay the moves and compare with the stored outcome
    game = tak_engine.Tak(komi=komi)
    for r in raws:
        game.make_move_raw(r)
    if game.result() != outcome:
        print(f"WARNING: replay result {game.result()} != stored outcome {outcome}")

    result = RESULT_TAGS.get(outcome, "0-0")
    headers = [
        ("Site", "self-play"),
        ("Event", f"self-play game {game_id}"),
        ("Player1", "white"),
        ("Player2", "black"),
        ("Size", "6"),
        ("Komi", str(komi)),
        ("Result", result),
        ("Seed", str(seed)),
        ("Model", str(model)),
        ("MaxSims", str(max_sims)),
        ("TempPlies", str(temp_plies)),
        ("EndReason", str(end_reason)),
    ]

    lines = [f'[{k} "{v}"]' for k, v in headers]
    lines.append("")
    for i in range(0, len(ptn_moves), 2):
        pair = ptn_moves[i:i + 2]
        lines.append(f"{i // 2 + 1}. " + " ".join(pair))
    lines.append(result)

    out_path = out_path or f"game_{game_id}.ptn"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(ptn_moves)} plies, result {result})")


if __name__ == "__main__":
    # gid = int(sys.argv[1]) if len(sys.argv) > 1 else GAME_ID
    # export_game(DB_PATH, gid, OUT_PATH)
    conn = sqlite3.connect(DB_PATH)
    print(conn.execute("SELECT COUNT(*) FROM self_play_games WHERE encoder_version = 1 ").fetchall())

