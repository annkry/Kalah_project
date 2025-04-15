# Implements Nested Rollout Policy Adaptation (NRPA), which learns a policy via softmax-weighted rollouts and adaptation.
# Parameters: level (nesting level), policy (dict of state-action weights), iterations (per level), root (starting node).

import random
from KalahGame import KalahGame
import math

# === NRPA (Nested Rollout Policy Adaptation) ===
# Uses softmax policy for rollouts, updated after each level
# P(s,a) ∝ exp(policy_score), adapted toward best sequences
def nrpa(level, policy, iterations, root):
    if level == 0:
        # base level does a rollout using the current policy
        return nrpa_rollout(root, policy)
    
    best_score = float('-inf')
    best_seq = []
    for _ in range(iterations):
        # do a rollout at a lower level
        sequence, score = nrpa(level - 1, policy.copy(), iterations, root)
        if score > best_score:
            best_score = score
            best_seq = sequence
        # update the policy to favor the best sequence
        policy = adapt(policy, best_seq)
    return best_seq, best_score

# simulates a game using softmax move selection from the policy
def nrpa_rollout(root, policy):
    game = root.get_game_copy()
    sequence = []

    while not game.is_terminal():
        moves = game.valid_moves(game.current_player)
        move_probs = softmax_probs(policy, game, moves)
        move = random.choices(moves, weights=move_probs)[0]
        sequence.append((move, game.from_board_to_state()))
        game.perform_move(move, game.current_player)

    # score the result of the game
    result = game.result()
    reward = 1 if result > 0 else -1 if result < 0 else 0
    return sequence, reward

# uses softmax to turn policy weights into probabilities
def softmax_probs(policy, game, moves):
    state = game.from_board_to_state()
    weights = [policy.get((state, move), 0) for move in moves]
    
    # subtract max for numerical stability
    max_weight = max(weights)
    exp_weights = [math.exp(w - max_weight) for w in weights]
    total = sum(exp_weights)

    # if something went wrong and total is 0, return uniform probs
    if total == 0:
        return [1 / len(moves)] * len(moves)
    
    return [w / total for w in exp_weights]

# updates the policy to make the given sequence more likely
def adapt(policy, sequence):
    for move, state in sequence:
        # create a new game state based on saved state info
        moves = KalahGame()
        moves.board = list(state[0])
        moves.current_player = state[1]
        valid_moves = moves.valid_moves(moves.current_player)
        probs = softmax_probs(policy, moves, valid_moves)

        # increase weight for chosen move, decrease for others
        for i, m in enumerate(valid_moves):
            key = (state, m)
            policy[key] = policy.get(key, 0)
            if m == move:
                policy[key] += 1
            else:
                policy[key] -= probs[i]
    return policy
