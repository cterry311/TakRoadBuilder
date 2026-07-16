from random import choice

import game



moves = [
'a6', 'f6',
'd4', 'c4',
'd3', 'c3',
'd5', 'Ce3',
'd6', 'c2',
'Cc5', 'd2',
'b4', 'e4',
'e6', 'a4',
'b5', 'a5',
'b3', '1c4<1',
'c4', '2b4-2',
'1c5<1', '1e4<1',
'c5', '2d4+2',
'2b5-11', 'e2',
'c6', '1c3+1',
'a3', '2c4+2',
'b5', 'b6',
'1b5>1', '3d5<3',
'1c6-1', 'Sc4',
'6c5<33', '1c4+1',
'd4', '1a4+1',
'1b5<1', '1a6-1',
'Sa4', '6a5+6',
'Sc6', 'd5',
'e5', 'f3',
'c4', '1d5-1',
'1d3+1', '3c5-3',
'3b3>12', '4c4-4',
'1d3<1', '1a5>1',
'4c3+4', '3b5-12',
'd5', '1c2+1',
'2c4-2', '1d3+1',
'1c4>1', '2b4>11',
'1c4>1', 'a1',
'4c3-13',
]

tak = game.Tak(6)

for move in moves:
    print(move)
    legal_moves = tak.get_all_legal_moves()
    if move in legal_moves:
        pass
    else:
        raise Exception('Invalid move')
    print(len(legal_moves))
    tak.make_move(move)
print(tak.turn)

# game_ptn = ''
#
# move_count = 0
#
# while True:
#     if move_count % 2 == 0:
#         game_ptn += '\n' + str((move_count // 2) + 1) + '.'
#     legal_moves = tak.get_all_legal_moves(filtering=3)
#     if len(legal_moves) == 0:
#         break
#     rand_move = choice(legal_moves)
#     game_ptn += ' ' + rand_move
#     print(rand_move)
#     print(len(legal_moves))
#     tak.make_move(rand_move)
#     move_count += 1
#     tak.display_board()
# print(tak.turn)
#
# with open('test2.ptn', 'w') as f:
#     f.write(game_ptn)

# tak = game.Tak(5)
#
#
#
# while True:
#     move = input('> ')
#     if move:
#         tak.make_move(move)
#     else:
#         moves = tak.get_all_legal_moves(filtering=2)
#         rand_move = choice(moves)
#         tak.make_move(rand_move)
#         print(rand_move)
#     if tak.turn in [4, 5, 6]:
#         print('Game over')
#         print(tak.turn)
#         break


# tak = game.Tak(5)
# tak.make_move('e1')
# tak.make_move('a1')
# tak.make_move('a2')
# tak.make_move('d1')
# tak.make_move('a3')
# tak.make_move('e2')
# tak.make_move('b4')
# moves = tak.get_all_legal_moves(filtering=3)
# print(moves)







# for i in range(1000):
#     tak = game.Tak(6)
#     while True:
#         game_copy = tak._copy()
#         legal_moves = tak.get_all_legal_moves()
#         for move in legal_moves:
#             tak.make_move(move)
#             tak.undo_move()
#             if game_copy == tak:
#                 pass
#             else:
#                 print('Failed Move: ' + move)
#                 print()
#                 print('Initial board: ' + str(game_copy.board))
#                 print('Resulting board: ' + str(tak.board))
#                 print('Board matches: ' + str(game_copy.board == tak.board))
#                 print()
#                 print('white stones before: ' + str(game_copy.white_stones))
#                 print('white stones after: ' + str(tak.white_stones))
#                 print('white stone count matches: ' + str(game_copy.white_stones == tak.white_stones))
#                 print()
#                 print('black stones before: ' + str(game_copy.black_stones))
#                 print('black stones after: ' + str(tak.black_stones))
#                 print('black stone count matches: ' + str(game_copy.black_stones == tak.black_stones))
#                 print()
#                 print('white capstones before: ' + str(game_copy.white_capstones))
#                 print('white capstones after: ' + str(tak.white_capstones))
#                 print('white capstone count matches: ' + str(game_copy.white_capstones == tak.white_capstones))
#                 print()
#                 print('black capstones before: ' + str(game_copy.black_capstones))
#                 print('black capstones after: ' + str(tak.black_capstones))
#                 print('black capstone count matches: ' + str(game_copy.black_capstones == tak.black_capstones))
#                 print()
#                 print('turn before: ' + str(game_copy.turn))
#                 print('turn after: ' + str(tak.turn))
#                 print('turn matches: ' + str(game_copy.turn == tak.turn))
#                 print()
#                 print('history matches: ' + str(game_copy.move_history == tak.move_history))
#                 print()
#                 game_copy.make_move(move)
#                 print('Move Recorded: ' + str(game_copy.move_history[-1]))
#                 raise Exception('Move Reversal Failed')
#         tak.make_move(choice(legal_moves))
#         if tak.turn in [4, 5, 6]:
#             print('Game over')
#             print(tak.turn)
#             break



