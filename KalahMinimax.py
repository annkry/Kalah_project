# Implements the Minimax algorithm with alpha-beta pruning for move selection in Kalah.
# Parameters: depth (search depth), alpha/beta (pruning bounds), minimax_player (which player to optimize for).

from Kalahevaluate import evaluate
import copy

# === Minimax with Alpha-Beta Pruning ===
# V(s) = max/min over child nodes depending on player
# Prune when beta <= alpha
# Terminal score = heuristic evaluation
def minimax(game, depth, alpha, beta, minimax_player):
    """
        Returns a tuple (score, best_move) for the current game state.
    
        - game: the current game state.
        - depth: the maximum depth to search.
        - alpha: the best already explored option along the path to the root for the maximizer.
        - beta: the best already explored option along the path to the root for the minimizer.
        - minimax_player: the player for whom we are optimizing.
    """
    
    # stop if we are at depth limit or game is over
    if depth == 0 or game.is_terminal():
        return evaluate(game, minimax_player), None

    current_player = game.current_player
    best_move = None

    if current_player == minimax_player:
        # this player is trying to maximize score
        max_eval = -float('inf')
        for move in game.valid_moves(current_player):
            new_game = copy.deepcopy(game)
            new_game.perform_move(move, current_player)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, minimax_player)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                # prune the rest if it is not worth continuing
                break
        return max_eval, best_move
    else:
        # this player is minimizing score (opponent)
        min_eval = float('inf')
        for move in game.valid_moves(current_player):
            new_game = copy.deepcopy(game)
            new_game.perform_move(move, current_player)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, minimax_player)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                # pruning here too
                break
        return min_eval, best_move

# simple wrapper function to make a minimax move
def minimax_move(game, depth = 2):
    """
        Returns the best move for the current minimax player
        using minimax search with alpha-beta pruning.
    
        The search depth can be adjusted via the depth parameter.
    """
    
    minimax_player = game.current_player  
    _, best_move = minimax(game, depth, -float('inf'), float('inf'), minimax_player)
    return best_move