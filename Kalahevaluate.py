# Contains heuristic evaluation functions for Kalah board states.
# Weights: weight1 = store difference, weight2 = extra move potential, weight3 = capture potential.

import copy

def evaluate(game, player, weight1 = 1.0, weight2 = 0.7, weight3 = 0.2):
    """
        Heuristic evaluation function for Kalah for game states.
    
        Factors:
        - Store difference: seeds in player's store minus seeds in opponent's store.
        - Extra move potential: count of moves that result in an extra turn.
        - Capture potential: count of moves that could result in a capture.
    """
    opponent = 1 - player

    # we get how many seeds each player has in their store
    store_diff = game.get_store(player) - game.get_store(opponent)

    # we simulate each move to see how many would give another turn
    extra_moves_player = count_extra_moves(game, player)
    extra_moves_opponent = count_extra_moves(game, opponent)
    extra_moves_diff = extra_moves_player - extra_moves_opponent

    # same for potential captures
    captures_player = count_potential_captures(game, player)
    captures_opponent = count_potential_captures(game, opponent)
    capture_diff = captures_player - captures_opponent

    # final score is just a weighted sum of all 3 parts
    return weight1 * store_diff + weight2 * extra_moves_diff + weight3 * capture_diff

def count_extra_moves(game, player):
    """
        Returns the number of moves for 'player' that would result in an extra turn.
        This function simulates each valid move to see if the last stone lands in the player's store.
    """
    extra_moves = 0
    for move in game.valid_moves(player):
        # copy the game to simulate without changing the real one
        new_game = copy.deepcopy(game)
        new_game.perform_move(move, player)
        # if the current player remains the same, an extra move was earned
        if new_game.current_player == player:
            extra_moves += 1
    return extra_moves

def count_potential_captures(game, player):
    """
        Returns the number of moves for 'player' that could potentially result in a capture.
        A capture occurs when the last stone lands in an empty pit on the player's side,
        and the opposite pit has seeds.
    """
    potential_captures = 0
    for move in game.valid_moves(player):
        # we create a copy to simulate the move
        new_game = copy.deepcopy(game)
        new_game.perform_move(move, player)
        ending_pit = new_game.last_pit
        if new_game.is_on_player_side(ending_pit, player) and new_game.get_seeds(ending_pit) == 1:
            opposite_pit = new_game.get_opposite_pit(ending_pit)
            if new_game.get_seeds(opposite_pit) > 0:
                potential_captures += 1
    return potential_captures