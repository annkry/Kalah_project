from collections import defaultdict
from KalahGame import KalahGame
from KalahMCTS import KalahMCTS
from KalahMCTS import MCTSNode
from Kalahevaluate import evaluate
import matplotlib.pyplot as plt
import random
import os

from NRPA import nrpa
from NMCS import nmcs
from UNMINIMAX import UBFM

# this function sets up and plays a match between two configs (agents)
# returns the net win difference (how much config1 wins more than config2)
def play_match(config1, config2, games = 2):
    wins = [0, 0]  # keep track of wins for each side

    for _ in range(games):
        game = KalahGame()

        # randomly decide who starts
        if random.random() < 0.5:
            config_first, config_second = config1, config2
        else:
            config_first, config_second = config2, config1

        # creates a player function depending on the config
        def create_player(config):
            if config.get("use_nrpa"):
                return lambda g: nrpa(
                    config.get("level", 1),
                    {},
                    config.get("iterations", 1000),
                    MCTSNode(g.from_board_to_state(), player = g.current_player)
                )[0][0]
            elif config.get("use_unbounded_minimax"):
                ubfm_agent = UBFM(evaluate)
                return lambda g: ubfm_agent.run(g, g.current_player, time_limit_seconds = 1.0)
            elif config.get("use_nmcs"):
                return lambda g: nmcs(
                    g,
                    level=config.get("nmcs_level", 3),
                    discounting=config.get("discounting", True),
                    prune_on_depth=config.get("prune_on_depth", True),
                    cut_on_win=config.get("cut_on_win", True)
                )[0]
            else:
                return lambda g: KalahMCTS(g, **config).best_move(g)

        players = [create_player(config_first), create_player(config_second)]

        # simulate the game until it ends
        while not game.is_terminal():
            current = game.current_player
            move = players[current](game)
            game.perform_move(move, current)

        result = game.result()
        # decide who actually won and update wins
        if result > 0:
            wins[0 if config_first == config1 else 1] += 1
        elif result < 0:
            wins[1 if config_first == config1 else 0] += 1


    return wins[0] - wins[1]

# this function tries out different parameter values and logs which ones perform best
def run_parameter_tuning():
    # these are the methods and parameters we are going to tune
    method_param_groups = {
        "nmcs": {
            "use_nmcs": [True],
            "nmcs_level": [1, 2, 3],
            "discounting": [True, False],
            "prune_on_depth": [True, False],
            "cut_on_win": [True, False]
        },
        "puct": {
            "use_puct": [True],
            "c_puct": [0.5, 1.0, 2.0]
        },
        "grave": {
            "use_grave": [True],
            "beta_const": [100, 300, 500],
            "threshold": [30, 50, 100]
        },
        "rave": {
            "use_rave": [True],
            "beta_const": [100, 300, 500],
        },
        "mcts_uct": {
            "c_param": [0.5, 1.0, 1.4, 2.0]
        },
        "shuss": {
            "use_shuss": [True],
            "shuss_c": [128.0, 256.0, 512.0]
        }
    }

    # base configuration
    default_config = {
        "iterations": 10000,
        "use_rave": False,
        "use_grave": False,
        "use_puct": False,
        "use_shot": False,
        "use_shuss": False,
        "c_param": 1.4,
        "beta_const": 300,
        "c_puct": 1.0,
        "shuss_c": 128.0
    }

    os.makedirs("plots", exist_ok = True)

    # tune each method separately
    for method, params in method_param_groups.items():
        print(f"\nTuning method: {method}")
        base_config = default_config.copy()

        for key in params:
            if key.startswith("use_"):
                base_config[key] = True  # enable the method

        best_values = {}
        tuning_history = defaultdict(list)

        for param, values in params.items():
            if param.startswith("use_"):
                continue # we skip the flags here

            best_score = -float('inf')
            best_value = None

            for value in sorted(values, key = lambda v: str(v)):
                test_config = base_config.copy()
                test_config[param] = value

                # actually play matches and evaluate
                score = play_match(test_config, base_config, games = 1000)
                print(f"  {param} = {value} => Net wins: {score}")

                tuning_history[param].append((value, score))

                if score > best_score:
                    best_score = score
                    best_value = value

            # after testing all values, save the best one
            base_config[param] = best_value
            best_values[param] = (best_value, best_score)

        print(f"\nBest values for method '{method}':")
        for param, (value, score) in best_values.items():
            print(f"  {param}: {value} (Net wins: {score})")

        # plot the performance for each param value
        plt.figure(figsize = (12, 6))
        all_labels = []
        label_map = {}
        idx = 0
        for param, values in tuning_history.items():
            for val, _ in values:
                label = f"{param}={val}"
                if label not in label_map:
                    label_map[label] = idx
                    all_labels.append(label)
                    idx += 1

        cmap = plt.colormaps.get_cmap('tab10')  # Get the colormap object
        colors = [cmap(i) for i in range(len(tuning_history))] 

        for i, (param, values) in enumerate(tuning_history.items()):
            x = [label_map[f"{param}={val}"] for val, _ in values]
            y = [score for _, score in values]
            plt.plot(x, y, marker='o', label=param, color=colors[i])

        plt.xticks(ticks=range(len(all_labels)), labels = all_labels, rotation = 45, ha = 'right')
        plt.title(f"Tuning results for {method.upper()}", fontsize = 32)
        plt.xlabel("Parameter = value", fontsize = 32)
        plt.ylabel("Net wins", fontsize = 32)
        plt.tick_params(axis='x', labelsize = 24)
        plt.tick_params(axis='y', labelsize = 24)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{method}_tuning.pdf")
        plt.close()

if __name__ == "__main__":
    run_parameter_tuning()