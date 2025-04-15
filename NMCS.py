# Implements Nested Monte Carlo Search (NMCS), where higher levels use deeper nested simulations.
# Parameters: level (depth of nesting), discounting (reward adjustment by depth), prune_on_depth, cut_on_win (early exits).

import random

# === NMCS (Nested Monte Carlo Search) ===
# At level N, test all actions with level N-1 rollouts
# Picks the move leading to the best result
def nested(game, level, depth = 1, bound = None, discounting = True, prune_on_depth = True, cut_on_win = True):
    if game.is_terminal():
        # if the game is over, just evaluate the result
        return None, evaluate_terminal(game, discounting, depth)

    player = game.current_player
    best_move = None
    best_value = -float('inf') if player == 0 else float('inf')
    values = {}

    # get all valid moves and shuffle to add randomness
    moves = game.valid_moves(player)
    random.shuffle(moves)

    for move in moves:
        # simulate move
        g_copy = game.get_game_copy()
        g_copy.perform_move(move, player)

        if level == 0:
            # base level just evaluates the terminal result of the rollout
            val = evaluate_terminal(g_copy, discounting, depth + 1)
        else:
            # do a nested call with one less level
            _, val = nested(g_copy, level - 1, depth + 1, best_value if prune_on_depth else None,
                            discounting, prune_on_depth, cut_on_win)

        values[move] = val

        # update best move and value depending on player
        if (player == 0 and val > best_value) or (player == 1 and val < best_value):
            best_value = val
            best_move = move

            # optional early exit if we found a guaranteed win
            if cut_on_win and ((player == 0 and val == 1) or (player == 1 and val == -1)):
                break

        # pruning step if enabled
        if prune_on_depth and discounting and bound is not None:
            if (player == 0 and val <= bound) or (player == 1 and val >= bound):
                continue

    return best_move, best_value

# simple utility to evaluate the result of a finished game
def evaluate_terminal(game, discounting, depth):
    result = game.result()
    if result > 0:
        value = 1
    elif result < 0:
        value = -1
    else:
        value = 0
    return value / depth if discounting else value

# this is the version of the NMCS function that the rest of the project calls
def nmcs(game, level = 1, discounting = True, prune_on_depth = True, cut_on_win = True):
    move, _ = nested(game, level, discounting = discounting, prune_on_depth = prune_on_depth, cut_on_win = cut_on_win)
    return move, _
