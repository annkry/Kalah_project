# Runs a grid search to tune heuristic evaluation weights by pitting greedy against random player.
# Parameters: w1 (store diff), w2 (extra move weight), w3 (capture weight).

from KalahGame import KalahGame
from Kalahevaluate import evaluate
from comparison_metrics import random_move

# === Greedy Search with Heuristic ===
# greedy player that picks the move with the best eval score based on the given weights
# a* = argmax_a evaluate(game_after_a)
# Heuristic: w1 * store_diff + w2 * extra_turns + w3 * captures
def greedy_move(game, w1, w2, w3):
    curr_player = game.current_player
    best_move, best_score = None, float('-inf')
    for move in game.valid_moves(curr_player):
        temp_game = game.get_game_copy()
        temp_game.perform_move(move, curr_player)
        score = evaluate(temp_game, curr_player, weight1 = w1, weight2 = w2, weight3 = w3)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

# this plays one game between a greedy player and a random player
# greedy is always player 0 and random is player 1
def play_game(w1, w2, w3):
    game = KalahGame()
    while not game.is_terminal():
        current = game.current_player
        move = greedy_move(game, w1, w2, w3) if current == 0 else random_move(game)
        game.perform_move(move, current)
    return game.result()

# weight combinations to test out
extra_weights = [0.3, 0.5, 0.7]
capture_weights = [0.2, 0.3, 0.5]

print("Tuning evaluation weights over 10 000 games per combination...\n")

# loop over all combos of extra and capture weights
for w2 in extra_weights:
    for w3 in capture_weights:
        wins = 0
        for _ in range(10000):
            result = play_game(1.0, w2, w3)
            if result > 0:
                wins += 1
        # print how many games greedy player won
        print(f"Extra Weight: {w2:.1f}, Capture Weight: {w3:.1f} -> Greedy Wins: {wins}/10000")

# OUTPUT:
# Extra Weight: 0.3, Capture Weight: 0.2 -> Greedy Wins: 9817/10000
# Extra Weight: 0.3, Capture Weight: 0.3 -> Greedy Wins: 9841/10000
# Extra Weight: 0.3, Capture Weight: 0.5 -> Greedy Wins: 9827/10000
# Extra Weight: 0.5, Capture Weight: 0.2 -> Greedy Wins: 9841/10000
# Extra Weight: 0.5, Capture Weight: 0.3 -> Greedy Wins: 9834/10000
# Extra Weight: 0.5, Capture Weight: 0.5 -> Greedy Wins: 9844/10000
# Extra Weight: 0.7, Capture Weight: 0.2 -> Greedy Wins: 9870/10000 <----- the best configuration
# Extra Weight: 0.7, Capture Weight: 0.3 -> Greedy Wins: 9845/10000
# Extra Weight: 0.7, Capture Weight: 0.5 -> Greedy Wins: 9831/10000