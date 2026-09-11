from game import Tak
from abc import ABC, abstractmethod
import random

class TakBot(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_move(self, game : Tak, do_move=True, curve_power=None):
        pass


class BruteBot(TakBot):

    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def make_move(self, game : Tak, do_move=True, curve_power=None):
        possible_moves = game.get_all_legal_moves(filtering=self.depth)
        chosen = random.choice(possible_moves)
        if do_move:
            game.make_move(chosen)
        return chosen


class EvalBot(TakBot):

    def __init__(self, depth, flat_weight=1, road_tile_weight=1, capstone_weight=0.5, control_weight=1, imprisoned_weight=1, self_control_weight=1, wall_weight=1):
        super().__init__()
        self.depth = depth
        self.flat_weight = flat_weight
        self.road_tile_weight = road_tile_weight
        self.capstone_weight = capstone_weight
        self.control_weight = control_weight
        self.imprisoned_weight = imprisoned_weight
        self.self_control_weight = self_control_weight
        self.wall_weight = wall_weight

    def eval(self, move : str, game : Tak):
        if game.turn in [0, 2]:
            own_flat = 0
            enemy_flat = 1
            own_road_tiles = [0, 4]
            enemy_road_tiles = [1, 5]
            own_tiles = [0, 2, 4]
            enemy_tiles = [1, 3, 5]
            own_wall = 2
            enemy_wall = 3
            own_capstone = 4
        else:
            own_flat = 1
            enemy_flat = 0
            own_road_tiles = [1, 5]
            enemy_road_tiles = [0, 4]
            own_tiles = [1, 3, 5]
            enemy_tiles = [0, 2, 4]
            own_wall = 3
            enemy_wall = 2
            own_capstone = 5
        game.make_move(move)
        board = game.board
        flat_diff = 0
        road_tile_diff = 0
        capstone_height = -2
        control_diff = 0
        imprisoned_diff = 0
        self_control_diff = 0
        wall_diff = 0
        for row in board:
            for stack in row:
                if stack:
                    stone = stack[-1] # get top stone

                    if stone == own_flat: # handle flat count differential
                        flat_diff += 1
                    elif stone == enemy_flat:
                        flat_diff -= 1

                    if stone in own_road_tiles: # handle road tile differential
                        road_tile_diff += 1
                    elif stone in enemy_road_tiles:
                        road_tile_diff -= 1

                    if stone == own_capstone: # handle capstone height
                        capstone_height = len(stack) - 1

                    if stone in own_tiles: # handle control differential
                        control_diff += 1
                    else:
                        control_diff -= 1

                    if stone == own_wall:
                        wall_diff += 1
                    elif stone == enemy_wall:
                        wall_diff -= 1


                    if len(stack) > 1:
                        under_stack = stack[:-1]
                        own_count = 0
                        enemy_count = 0
                        for stone_under in under_stack:
                            if stone_under == own_flat:
                                own_count += 1
                            else:
                                enemy_count += 1
                        if stone in own_tiles:
                            imprisoned_diff += enemy_count
                            self_control_diff += own_count
                        else:
                            imprisoned_diff += own_count
                            self_control_diff += enemy_count
        score = (flat_diff * self.flat_weight) + (road_tile_diff * self.road_tile_weight) + (capstone_height * self.capstone_weight) + (control_diff * self.control_weight) + (imprisoned_diff * self.imprisoned_weight) + (self_control_diff * self.self_control_weight) + (wall_diff * self.wall_weight)
        game.undo_move()
        return score

    def make_move(self, game : Tak, do_move=True, curve_power=None):
        possible_moves = game.get_all_legal_moves(filtering=self.depth)
        scores = []
        for move in possible_moves:
            scores.append((self.eval(move, game), move))
        if curve_power is not None:
            scores_ajusted = []
            min_score = float('inf')
            for score, _ in scores:
                if score < min_score:
                    min_score = score
            for score, _ in scores:
                scores_ajusted.append(((score - min_score) + 1) ** curve_power)
            ajusted_total = sum(scores_ajusted)
            chosen_value = random.uniform(0, ajusted_total)
            current_total = 0
            chosen_index = 0
            for i, score in enumerate(scores_ajusted):
                current_total += score
                if current_total >= chosen_value:
                    chosen_index = i
                    break
            move_meta_score = (scores_ajusted[chosen_index] / ajusted_total) * len(scores_ajusted)
            print()
            print(f'Meta Move Score: {move_meta_score}')
            chosen = possible_moves[chosen_index]
        else:
            best_moves = []
            for score, move in scores:
                if not best_moves or score > best_moves[0][0]:
                    best_moves = [(score, move)]
                elif score == best_moves[0][0]:
                    best_moves.append((score, move))
            chosen = random.choice(best_moves)[1]
        if do_move:
            game.make_move(chosen)
        return chosen