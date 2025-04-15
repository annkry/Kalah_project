# Runs tournament matches between different strategies, logs detailed metrics and saves results.
# Parameters: num_games (games per match), strategy names, timing tracking per player.

import matplotlib.pyplot as plt
import time
import random
import csv
from KalahGame import KalahGame
from KalahMCTS import KalahMCTS
from KalahMCTS import MCTSNode
from KalahMinimax import minimax_move
from Kalahevaluate import evaluate, count_extra_moves

from NRPA import nrpa
from NMCS import nmcs
from UNMINIMAX import UBFM

# this function picks a random move from the valid ones
def random_move(game):
    return random.choice(game.valid_moves(game.current_player))

# plays a match between two strategies for a bunch of games and collects stats
def play_match(strategy1_name, strategy2_name, num_games = 100):
    wins = {strategy1_name: 0, strategy2_name: 0}
    metrics = {
        "moves": [],
        "score_diff": [],
        "extra_turns": {strategy1_name: [], strategy2_name: []},
        "time_taken": {strategy1_name: [], strategy2_name: []}
    }
    print(f"Playing {num_games} games between {strategy1_name} and {strategy2_name}...")

    for _ in range(num_games):
        game = KalahGame()

        # set up agents for each strategy
        mcts = KalahMCTS(game, iterations = 10000, c_param = 1.4)
        rave = KalahMCTS(game, iterations = 10000, use_rave = True, beta_const = 300)
        grave = KalahMCTS(game, iterations = 10000, use_grave = True, beta_const = 300, threshold = 30)
        puct = KalahMCTS(game, iterations = 10000, use_puct = True, c_puct = 2.0)
        shot = KalahMCTS(game, iterations = 10000, use_shot = True)
        shuss = KalahMCTS(game, iterations = 10000, use_shuss = True, shuss_c = 128.0)

        # all strategies in one place
        players = {
            "random": random_move,
            "MCTS": lambda g: mcts.best_move(g),
            "Minimax": lambda g: minimax_move(g, depth = 2),
            "RAVE": lambda g: rave.best_move(g),
            "GRAVE": lambda g: grave.best_move(g),
            "NRPA": lambda g: nrpa(
                    1,
                    {},
                    1000,
                    MCTSNode(g.from_board_to_state(), player = g.current_player)
                )[0][0][0],
            "PUCT": lambda g: puct.best_move(g),
            "SHOT": lambda g: shot.best_move(g),
            "NMCS": lambda g: nmcs(
                    g,
                    level = 3,
                    discounting = False,
                    prune_on_depth = False,
                    cut_on_win = False
                )[0],
            "Unbounded Minimax": lambda g: UBFM(evaluate).run(g, g.current_player, time_limit_seconds = 1.0),
            "SHUSS": lambda g: shuss.best_move(g),
        }

        # randomize who goes first
        if random.random() < 0.5:
            player0_name, player1_name = strategy1_name, strategy2_name
        else:
            player0_name, player1_name = strategy2_name, strategy1_name

        player0 = players[player0_name]
        player1 = players[player1_name]

        move_count, p1_time, p2_time = 0, 0.0, 0.0
        while not game.is_terminal():
            if game.valid_moves(0) == [None] and game.valid_moves(1) == [None]:
                break
            move_count += 1
            current = game.current_player
            start_time = time.time()
            move = player0(game) if current == 0 else player1(game)
            elapsed = time.time() - start_time
            game.perform_move(move, current)
            if current == 0:
                p1_time += elapsed
            else:
                p2_time += elapsed

        result = game.result()
        # update the win counter depending on result
        if result > 0:
            wins[player0_name] += 1
        else:
            wins[player1_name] += 1

        # collect game stats
        metrics["moves"].append(move_count)
        metrics["score_diff"].append(result)
        metrics["extra_turns"][player0_name].append(count_extra_moves(game, 0))
        metrics["extra_turns"][player1_name].append(count_extra_moves(game, 1))
        metrics["time_taken"][player0_name].append(p1_time)
        metrics["time_taken"][player1_name].append(p2_time)

    win_strategy = strategy1_name if wins[strategy1_name] > wins[strategy2_name] else strategy2_name
    ratio = wins[win_strategy] / num_games
    avg_metrics = {
        "avg_moves": sum(metrics["moves"]) / num_games,
        "avg_score_diff": sum(abs(x) for x in metrics["score_diff"]) / num_games,
        strategy1_name + "_avg_extra": sum(metrics["extra_turns"][strategy1_name]) / num_games,
        strategy2_name + "_avg_extra": sum(metrics["extra_turns"][strategy2_name]) / num_games,
        strategy1_name + "_avg_time": sum(metrics["time_taken"][strategy1_name]) / num_games,
        strategy2_name + "_avg_time": sum(metrics["time_taken"][strategy2_name]) / num_games,
    }
    return (strategy1_name, strategy2_name, ratio, win_strategy, avg_metrics)

# tournament format where the winner keeps advancing in the list
def tournament_ladder(strategies):
    results = []
    current_winner = strategies[0]
    for i in range(1, len(strategies)):
        match_result = play_match(current_winner, strategies[i])
        results.append(match_result)
        current_winner = match_result[3]
    return results

# this saves the tournament results to a csv file
def save_results_to_csv(results, filename="tournament_results.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Match", "Winner", "Win ratio", "Avg moves", "Avg score diff",
            "P1 extra", "P2 extra", "P1 time (s)", "P2 time (s)"
        ])
        for s1, s2, ratio, winner, metrics in results:
            writer.writerow([
                f"{s1} vs {s2}", winner, ratio,
                metrics["avg_moves"], metrics["avg_score_diff"],
                metrics[s1 + "_avg_extra"], metrics[s2 + "_avg_extra"],
                metrics[s1 + "_avg_time"], metrics[s2 + "_avg_time"]
            ])

# draws a nice horizontal bar chart of all matches
def plot_tournament(results):
    fig, ax = plt.subplots(figsize = (12, 6))
    y_pos = list(range(len(results)))

    for idx, (s1, s2, ratio, win, _) in enumerate(results):
        ax.barh(idx, ratio, color='green')
        ax.barh(idx, 1 - ratio, left=ratio, color='red')
        ax.text(
                1.01, idx,
                f"{s1} vs {s2} → {win} wins with {ratio:.2f} ",
                va='center', fontsize=8
            )


    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s1} vs {s2}" for s1, s2, *_ in results])
    ax.set_xlabel("Win ratio")
    ax.set_title("Kalah tournament")
    ax.margins(x=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # list of all strategies to run in the ladder
    strategies = [
        "random", "MCTS", "RAVE", "GRAVE", "NRPA", "PUCT", "SHOT", "Unbounded Minimax", "SHUSS", "NMCS",  "Minimax"
    ]
    results = tournament_ladder(strategies)
    save_results_to_csv(results, "tournament_results.csv")
    plot_tournament(results)
