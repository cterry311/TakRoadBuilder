import copy
from collections import deque

_sizes = {
    3: [10, 1],
    4: [15, 0],
    5: [21, 1],
    6: [30, 1],
    7: [40, 2],
    8: [50, 2],
}



class Tak:

    def __init__(self, board_size: int):
        self.turn_states = ['white_first', 'black_first', 'white', 'black', 'draw', 'white_win', 'black_win']
        self.piece_types = ['white_flat', 'black_flat', 'white_wall', 'black_wall', 'white_capstone', 'black_capstone']
        self.board_size = board_size
        self.board = []
        self.white_stones = _sizes[board_size][0]
        self.black_stones = _sizes[board_size][0]
        self.white_capstones = _sizes[board_size][1]
        self.black_capstones = _sizes[board_size][1]
        self.turn = 0
        self.move_history = []
        for i in range(board_size):
            row = []
            for j in range(board_size):
                row.append([])
            self.board.append(row)

    def _handle_flat_win(self, simplified_board):
        white_count = 0
        black_count = 0
        for row in simplified_board:
            for stone in row:
                if stone == 0:
                    white_count += 1
                elif stone == 1:
                    black_count += 1
        if white_count > black_count:
            self.turn = 5
        if black_count > white_count:
            self.turn = 6
        if white_count == black_count:
            self.turn = 4

    def _find_road_win(self, simplified_board):
        # Returns 0 if no road win, 1 if white wins, -1 if black wins
        # If both have roads (possible via stack movement), the current mover wins
        # Road pieces: white = 0, 4 (flat, capstone) | black = 1, 5 (flat, capstone)

        WHITE_ROAD_PIECES = {0, 4}
        BLACK_ROAD_PIECES = {1, 5}

        def bfs_has_road(road_pieces):
            # For a road, we need to connect either:
            # top/bottom edges (rows 0 and board_size-1)
            # OR left/right edges (cols 0 and board_size-1)

            visited = set()

            # Try top-to-bottom road
            queue = []
            for col in range(self.board_size):
                if simplified_board[0][col] in road_pieces:
                    queue.append((0, col))
                    visited.add((0, col))

            while queue:
                row, col = queue.pop(0)
                if row == self.board_size - 1:
                    return True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.board_size and
                            0 <= nc < self.board_size and
                            (nr, nc) not in visited and
                            simplified_board[nr][nc] in road_pieces):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            visited = set()

            # Try left-to-right road
            queue = []
            for row in range(self.board_size):
                if simplified_board[row][0] in road_pieces:
                    queue.append((row, 0))
                    visited.add((row, 0))

            while queue:
                row, col = queue.pop(0)
                if col == self.board_size - 1:
                    return True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.board_size and
                            0 <= nc < self.board_size and
                            (nr, nc) not in visited and
                            simplified_board[nr][nc] in road_pieces):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            return False

        white_has_road = bfs_has_road(WHITE_ROAD_PIECES)
        black_has_road = bfs_has_road(BLACK_ROAD_PIECES)

        if white_has_road and black_has_road:
            # Current mover wins the tiebreak
            # turns 0,2 = white moving | turns 1,3 = black moving
            if self.turn in [0, 2]:
                return 1
            else:
                return -1
        if white_has_road:
            return 1
        if black_has_road:
            return -1
        return 0

    def display_board(self):
        simplified_board = []
        for i in range(self.board_size):
            simplified_board.append([])
            for j in range(self.board_size):
                top = -1
                if self.board[i][j]:
                    top = self.board[i][j][-1]
                simplified_board[i].append(top)
        for row in simplified_board:
            for stone in row:
                if stone == -1:
                    print('* ', end='')
                else:
                    print(str(stone) + ' ', end='')
            print()

    def _advance_turn(self):
        simplified_board = []
        for i in range(self.board_size):
            simplified_board.append([])
            for j in range(self.board_size):
                top = -1
                if self.board[i][j]:
                    top = self.board[i][j][-1]
                simplified_board[i].append(top)
        road_result = self._find_road_win(simplified_board)
        if road_result != 0:
            if road_result == 1:
                self.turn = 5
            if road_result == -1:
                self.turn = 6
            return
        if (self.white_stones == 0) or (self.black_stones == 0):
            self._handle_flat_win(simplified_board)
            return
        empty_count = 0
        for row in simplified_board:
            for stone in row:
                if stone == -1:
                    empty_count += 1
        if empty_count == 0:
            self._handle_flat_win(simplified_board)
            return
        if self.turn == 0:
            self.turn = 1
            return
        if self.turn == 1:
            self.turn = 2
            return
        if self.turn == 2:
            self.turn = 3
            return
        if self.turn == 3:
            self.turn = 2
            return


    def make_move(self, move: str):
        numbers = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
        letters = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        initial_turn = self.turn
        is_movement = False
        if self.turn in [4, 5, 6]:
            raise ValueError('game is already finished ' + move)
        for char in move:
            if char in ['+', '-', '<', '>']:
                is_movement = True
        if is_movement and self.turn in [0, 1]:
            raise ValueError('invalid move for first turn ' + move)
        if not is_movement:
            if not (len(move) in [2, 3]):
                raise ValueError('Invalid PTN format  ' + move)
            if move[0] == 'S':
                if self.turn in [0, 1]:
                    raise ValueError('invalid move for first turn Placing Wall' + move)
                position_x = letters[move[1]]
                position_y = numbers[move[2]]
                if self.turn == 2:
                    if self.white_stones > 0:
                        if not self.board[position_y][position_x]:
                            self.white_stones -= 1
                            self.board[position_y][position_x].append(2)
                            self._advance_turn()
                            moveInfo = {
                                'type': 'place_stone',
                                'position': (position_x, position_y),
                                'stone_type': 2,
                                'prev_turn': initial_turn
                            }
                            self.move_history.append(moveInfo)
                            return
                        else:
                            raise ValueError('Space is already occupied  ' + move)
                    else:
                        raise ValueError('No more stones available  ' + move)
                if self.turn == 3:
                    if self.black_stones > 0:
                        if not self.board[position_y][position_x]:
                            self.black_stones -= 1
                            self.board[position_y][position_x].append(3)
                            self._advance_turn()
                            moveInfo = {
                                'type': 'place_stone',
                                'position': (position_x, position_y),
                                'stone_type': 3,
                                'prev_turn': initial_turn
                            }
                            self.move_history.append(moveInfo)
                            return
                        else:
                            raise ValueError('Space is already occupied  ' + move)
                    else:
                        raise ValueError('No more stones available  ' + move)
                raise ValueError('game is already finished ' + move)
            if move[0] == 'C':
                if self.turn in [0, 1]:
                    raise ValueError('invalid move for first turn Placing Capstone' + move)
                position_x = letters[move[1]]
                position_y = numbers[move[2]]
                if self.turn == 2:
                    if self.white_capstones > 0:
                        if not self.board[position_y][position_x]:
                            self.white_capstones -= 1
                            self.board[position_y][position_x].append(4)
                            self._advance_turn()
                            moveInfo = {
                                'type': 'place_stone',
                                'position': (position_x, position_y),
                                'stone_type': 4,
                                'prev_turn': initial_turn
                            }
                            self.move_history.append(moveInfo)
                            return
                        else:
                            raise ValueError('Space is already occupied  ' + move)
                    else:
                        raise ValueError('No more capstones available  ' + move)
                if self.turn == 3:
                    if self.black_capstones > 0:
                        if not self.board[position_y][position_x]:
                            self.black_capstones -= 1
                            self.board[position_y][position_x].append(5)
                            self._advance_turn()
                            moveInfo = {
                                'type': 'place_stone',
                                'position': (position_x, position_y),
                                'stone_type': 5,
                                'prev_turn': initial_turn
                            }
                            self.move_history.append(moveInfo)
                            return
                        else:
                            raise ValueError('Space is already occupied  ' + move)
                    else:
                        raise ValueError('No more capstones available  ' + move)
                raise ValueError('game is already finished ' + move)
            if move[0] in letters.keys():
                position_x = letters[move[0]]
                position_y = numbers[move[1]]
                if self.turn in [1, 2]:
                    if self.white_stones > 0:
                        if not self.board[position_y][position_x]:
                            self.white_stones -= 1
                            self.board[position_y][position_x].append(0)
                            self._advance_turn()
                            moveInfo = {
                                'type': 'place_stone',
                                'position': (position_x, position_y),
                                'stone_type': 0,
                                'prev_turn': initial_turn
                            }
                            self.move_history.append(moveInfo)
                            return
                        else:
                            raise ValueError('Space is already occupied  ' + move)
                    else:
                        raise ValueError('No more stones available  ' + move)
                if self.turn in [0, 3]:
                    if self.black_stones > 0:
                        if not self.board[position_y][position_x]:
                            self.black_stones -= 1
                            self.board[position_y][position_x].append(1)
                            self._advance_turn()
                            moveInfo = {
                                'type': 'place_stone',
                                'position': (position_x, position_y),
                                'stone_type': 1,
                                'prev_turn': initial_turn
                            }
                            self.move_history.append(moveInfo)
                            return
                        else:
                            raise ValueError('Space is already occupied  ' + move)
                    else:
                        raise ValueError('No more stones available  ' + move)
                raise ValueError('game is already finished ' + move)
            raise ValueError('Invalid PTN format  ' + move)
        if is_movement:
            carry_amount = 0
            has_prefix = False
            if move[0] in numbers.keys():
                carry_amount = int(move[0])
                has_prefix = True
            else:
                carry_amount = 1
                has_prefix = False
            direction = ''
            if carry_amount > self.board_size:
                raise ValueError('Carry Limit exceeded  ' + move)
            if has_prefix:
                direction = move[3]
            else:
                direction = move[2]
            if not (direction in ['+', '-', '<', '>']):
                raise ValueError('Invalid PTN format  ' + move)
            split_move = move.split(direction)
            position_x = 0
            position_y = 0
            if len(split_move[0]) == 2:
                position_x = letters[split_move[0][0]]
                position_y = numbers[split_move[0][1]]
            elif len(split_move[0]) == 3:
                position_x = letters[split_move[0][1]]
                position_y = numbers[split_move[0][2]]
            if not self.board[position_y][position_x]:
                raise ValueError('No stack to move  ' + move)
            controlling_piece = self.board[position_y][position_x][-1]
            if not ((self.turn == 2 and controlling_piece in [0, 2, 4]) or (
                    self.turn == 3 and controlling_piece in [1, 3, 5])):
                raise ValueError('Cannot move this stack, not under your control  ' + move)
            carry_pattern = []
            if split_move[1] == '':
                carry_pattern.append(carry_amount)
            else:
                for char in split_move[1]:
                    carry_pattern.append(int(char))
            stones_to_move = []
            for i in range(carry_amount):
                stones_to_move.append(self.board[position_y][position_x].pop())
            x_modifier = 0
            y_modifier = 0
            if direction == '+':
                y_modifier = 1
            elif direction == '-':
                y_modifier = -1
            elif direction == '<':
                x_modifier = -1
            elif direction == '>':
                x_modifier = 1
            smashed_stone = False
            for i, amount in enumerate(carry_pattern):
                for j in range(amount):
                    stone_to_move = stones_to_move[-1]
                    current_stack = self.board[position_y + (i + 1) * y_modifier][position_x + (i + 1) * x_modifier]
                    if current_stack:
                        top_stone = current_stack[-1]
                        if top_stone in [2, 3]:
                            if stone_to_move in [4, 5]:
                                self.board[position_y + (i + 1) * y_modifier][position_x + (i + 1) * x_modifier][-1] -= 2
                                smashed_stone = True
                            else:
                                raise ValueError('Cannot move non capstone onto wall ' + move)
                        if top_stone in [4, 5]:
                            raise ValueError('Cannot onto capstone ' + move)
                    self.board[position_y + (i + 1) * y_modifier][position_x + (i + 1) * x_modifier].append(stones_to_move.pop())
            self._advance_turn()
            moveInfo = {
                'type': 'move_stack',
                'position': (position_x, position_y),
                'carry_pattern': carry_pattern,
                'direction': direction,
                'smashed_stone': smashed_stone,
                'prev_turn': initial_turn
            }
            self.move_history.append(moveInfo)
            return
        raise ValueError('Invalid PTN format  ' + move)


    def _enumerate_sequences(self, n : int, k : int, soft_stop : bool, first_step : bool) -> list:
        all_sequences = []
        if not first_step:
            all_sequences.append(())
        if n == 0:
            return all_sequences
        if k == 0:
            if soft_stop:
                all_sequences.append((1,))
            return all_sequences
        for i in range(1, n + 1):
            new_sequences = self._enumerate_sequences(n - i, k - 1, soft_stop, False)
            for seq in new_sequences:
                all_sequences.append((i,) + seq)
        return all_sequences


    def _handle_movement_enumeration(self, x : int, y : int):
        legal_moves = []
        stack_size = min(len(self.board[y][x]), self.board_size) # how many stones we have to work with in the drop pattern

        spaces_above = 0
        above_soft = False
        while True:
            if y - 1 - spaces_above < 0 or y - 1 - spaces_above >= self.board_size:
                break
            stack = self.board[y - 1 - spaces_above][x]
            if stack:
                if stack[-1] in [4, 5]:
                    break
                elif stack[-1] in [2, 3]:
                    if self.board[y][x][-1] in [4, 5]:
                        above_soft = True
                    break
            spaces_above += 1


        spaces_below = 0
        below_soft = False
        while True:
            if y + 1 + spaces_below < 0 or y + 1 + spaces_below >= self.board_size:
                break
            stack = self.board[y + 1 + spaces_below][x]
            if stack:
                if stack[-1] in [4, 5]:
                    break
                elif stack[-1] in [2, 3]:
                    if self.board[y][x][-1] in [4, 5]:
                        below_soft = True
                    break
            spaces_below += 1

        spaces_left = 0
        left_soft = False
        while True:
            if x - 1 - spaces_left < 0 or x - 1 - spaces_left >= self.board_size:
                break
            stack = self.board[y][x - 1 - spaces_left]
            if stack:
                if stack[-1] in [4, 5]:
                    break
                elif stack[-1] in [2, 3]:
                    if self.board[y][x][-1] in [4, 5]:
                        left_soft = True
                    break
            spaces_left += 1

        spaces_right = 0
        right_soft = False
        while True:
            if x + 1 + spaces_right < 0 or x + 1 + spaces_right >= self.board_size:
                break
            stack = self.board[y][x + 1 + spaces_right]
            if stack:
                if stack[-1] in [4, 5]:
                    break
                elif stack[-1] in [2, 3]:
                    if self.board[y][x][-1] in [4, 5]:
                        right_soft = True
                    break
            spaces_right += 1


        up_drop_patterns = self._enumerate_sequences(stack_size, spaces_above, above_soft, True)
        down_drop_patterns = self._enumerate_sequences(stack_size, spaces_below, below_soft, True)
        left_drop_patterns = self._enumerate_sequences(stack_size, spaces_left, left_soft, True)
        right_drop_patterns = self._enumerate_sequences(stack_size, spaces_right, right_soft, True)

        numbers = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'}
        letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

        move_cords = letters[x] + numbers[y]
        for drop_pattern in up_drop_patterns:
            moved_amount = sum(drop_pattern)
            move = str(moved_amount) + move_cords + '-'
            for amount in drop_pattern:
                move += str(amount)
            legal_moves.append(move)

        for drop_pattern in down_drop_patterns:
            moved_amount = sum(drop_pattern)
            move = str(moved_amount) + move_cords + '+'
            for amount in drop_pattern:
                move += str(amount)
            legal_moves.append(move)

        for drop_pattern in left_drop_patterns:
            moved_amount = sum(drop_pattern)
            move = str(moved_amount) + move_cords + '<'
            for amount in drop_pattern:
                move += str(amount)
            legal_moves.append(move)

        for drop_pattern in right_drop_patterns:
            moved_amount = sum(drop_pattern)
            move = str(moved_amount) + move_cords + '>'
            for amount in drop_pattern:
                move += str(amount)
            legal_moves.append(move)

        return legal_moves


    def _get_all_moves_base(self):
        numbers = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'}
        letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        legal_moves = []
        if self.turn in [4, 5, 6]:
            return legal_moves
        do_movement = self.turn in [2, 3]
        for y in range(self.board_size):
            for x in range(self.board_size):
                if not self.board[y][x]:
                    # handle stone placement
                    legal_moves.append(letters[x] + numbers[y])
                    if self.turn in [2, 3]:
                        # if not first turn, add walls
                        legal_moves.append('S' + letters[x] + numbers[y])
                        if (self.turn == 2 and self.white_capstones > 0) or (
                                self.turn == 3 and self.black_capstones > 0):
                            # if capstones available, add capstones
                            legal_moves.append('C' + letters[x] + numbers[y])
                elif do_movement:
                    # if doing movement, and there is at least 1 stone on the current stack, handle movement
                    if (self.turn == 2 and self.board[y][x][-1] in [0, 2, 4]) or (
                            self.turn == 3 and self.board[y][x][-1] in [1, 3, 5]):
                        # check if the stack is under control of the current mover
                        legal_moves.extend(self._handle_movement_enumeration(x, y))
        return legal_moves



    def get_all_legal_moves(self, filtering=-1):
        legal_moves = self._get_all_moves_base()
        if filtering == -1:
            return legal_moves
        return self._filter_recursive(legal_moves, include_result=False, remaining_depth=filtering)


    def _copy(self):
        new_game = Tak(self.board_size)
        new_game.turn = self.turn
        new_game.white_stones = self.white_stones
        new_game.black_stones = self.black_stones
        new_game.white_capstones = self.white_capstones
        new_game.black_capstones = self.black_capstones
        new_game.board = copy.deepcopy(self.board)
        new_game.move_history = copy.deepcopy(self.move_history)
        return new_game

    def undo_move(self):
        last_move = self.move_history.pop()
        self.turn = last_move['prev_turn']
        if last_move['type'] == 'place_stone':
            x, y = last_move['position']
            self.board[y][x].pop()
            match last_move['stone_type']:
                case 0:
                    self.white_stones += 1
                case 1:
                    self.black_stones += 1
                case 2:
                    self.white_stones += 1
                case 3:
                    self.black_stones += 1
                case 4:
                    self.white_capstones += 1
                case 5:
                    self.black_capstones += 1
        elif last_move['type'] == 'move_stack':
            x, y = last_move['position']
            x_modifier = 0
            y_modifier = 0
            if last_move['direction'] == '+':
                y_modifier = 1
            elif last_move['direction'] == '-':
                y_modifier = -1
            elif last_move['direction'] == '<':
                x_modifier = -1
            else:
                x_modifier = 1

            for i, count in enumerate(last_move['carry_pattern']):
                for j in range(count):
                    stone_to_move = self.board[y + ((i + 1) * y_modifier)][x + ((i + 1) * x_modifier)].pop(j - count)
                    self.board[y][x].append(stone_to_move)
            if last_move['smashed_stone']:
                self.board[y + (len(last_move['carry_pattern']) * y_modifier)][x + (len(last_move['carry_pattern']) * x_modifier)][-1] += 2





    def simulate_move(self, move):
        self.make_move(move)
        result = self.turn
        self.undo_move()
        return result

    def _filter_base(self, moves, include_result=False):
        winning_moves = []
        neutral_moves = []
        losing_moves = []
        for move in moves:
            result = self.simulate_move(move)
            if result in [0, 1, 2, 3, 4]:
                neutral_moves.append(move)
            elif self.turn == 2 and result == 5:
                winning_moves.append(move)
                break
            elif self.turn == 3 and result == 6:
                winning_moves.append(move)
                break
            else:
                losing_moves.append(move)
        if winning_moves:
            if include_result:
                return winning_moves, 1
            return winning_moves
        if neutral_moves:
            if include_result:
                return neutral_moves, 0
            return neutral_moves
        if include_result:
            return losing_moves, -1
        return losing_moves


    def _filter_recursive(self, moves, include_result=False, remaining_depth=0):
        level_1_filtered, result = self._filter_base(moves, include_result=True)
        if result == 1:
            if include_result:
                return level_1_filtered, 1
            return level_1_filtered
        if result == -1:
            if include_result:
                return level_1_filtered, -1
            return level_1_filtered
        if remaining_depth == 0:
            if include_result:
                return level_1_filtered, 0
            return level_1_filtered
        winning_moves = []
        neutral_moves = []
        losing_moves = []
        for move in level_1_filtered:
            self.make_move(move)
            opponent_moves = self.get_all_legal_moves()
            _, opponent_result = self._filter_recursive(opponent_moves, include_result=True, remaining_depth=remaining_depth - 1)
            self.undo_move()
            if opponent_result == 1:
                losing_moves.append(move)
            elif opponent_result == -1:
                winning_moves.append(move)
                break
            else:
                neutral_moves.append(move)
        if winning_moves:
            if include_result:
                return winning_moves, 1
            return winning_moves
        if neutral_moves:
            if include_result:
                return neutral_moves, 0
            return neutral_moves
        if include_result:
            return losing_moves, -1
        return losing_moves

    def __eq__(self, other):
        return self.board == other.board and self.turn == other.turn and self.white_stones == other.white_stones and self.black_stones == other.black_stones and self.white_capstones == other.white_capstones and self.black_capstones == other.black_capstones and self.move_history == other.move_history
