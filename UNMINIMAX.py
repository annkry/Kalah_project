# Implements Unbounded Minimax using a transposition table to iteratively improve move evaluations.
# Parameters: eval_fn (evaluation function), time_limit_seconds (allowed time per move).

import time

# === Unbounded Minimax (UBFM) ===
# Iteratively explores the best move using a transposition table
# V(s) = evaluated if new, refined recursively if visited
class UBFM:
    def __init__(self, eval_fn):
        # T is the transposition table (basically a memory)
        # it maps a state to all the actions from it and their values
        self.T = {}
        self.eval = eval_fn

    def best_action(self, state_key, player):
        # returns the best move based on stored values in T
        actions = list(self.T[state_key].keys())
        if player == 0:
            return max(actions, key=lambda a: self.T[state_key][a])
        else:
            return min(actions, key=lambda a: self.T[state_key][a])

    def unbounded_minimax_iteration(self, state, player):
        # runs a single unbounded minimax iteration

        if state.is_terminal():
            return self.eval(state, player)

        state_key = state.from_board_to_state()

        # if state hasn’t been explored, initialize its children
        if state_key not in self.T:
            self.T[state_key] = {}
            for action in state.valid_moves(state.current_player):
                s_prime = state.get_game_copy()
                s_prime.perform_move(action, s_prime.current_player)
                self.T[state_key][action] = self.eval(s_prime, player)
        else:
            # pick the best move so far and go deeper on that path
            best_a = self.best_action(state_key, player)
            s_prime = state.get_game_copy()
            s_prime.perform_move(best_a, s_prime.current_player)
            self.T[state_key][best_a] = self.unbounded_minimax_iteration(s_prime, player)

        best_a = self.best_action(state_key, player)
        return self.T[state_key][best_a]

    def run(self, root_game, player, time_limit_seconds = 1.0):
        # keeps running iterations until time runs out

        start = time.time()
        while time.time() - start < time_limit_seconds:
            self.unbounded_minimax_iteration(root_game, player)

        # once time is up, return the best move from root
        root_key = root_game.from_board_to_state()
        return self.best_action(root_key, player)
