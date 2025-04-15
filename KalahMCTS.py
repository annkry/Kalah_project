# Implements Monte Carlo Tree Search and its variants: RAVE, GRAVE, PUCT, SHOT, SHUSS.
# Parameters include c_param (exploration factor), beta_const (for RAVE/GRAVE), c_puct (for PUCT prior influence),
# shuss_c (for SHUSS), iterations (number of simulations), and threshold (min visits for GRAVE node activation).

import math
import random
from collections import defaultdict
from KalahGame import KalahGame
from Kalahevaluate import evaluate

class MCTSNode:
    def __init__(self, game_state, parent = None, move = None, player = 0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = self.get_valid_moves()
        self.player = player

        # for RAVE and GRAVE stats
        self.rave_visits = defaultdict(int)
        self.rave_value = defaultdict(float)

        # used when PUCT is enabled
        self.prior = 1.0

    def get_valid_moves(self):
        # get valid moves from the current state
        game = self.get_game_copy()
        return game.valid_moves(game.current_player)

    def get_game_copy(self):
        # recreates the game from stored state
        game = KalahGame()
        game.board = list(self.game_state[0])
        game.current_player = self.game_state[1]
        return game

    def is_fully_expanded(self):
        # returns true when there are no untried moves left
        return len(self.untried_moves) == 0

    def best_child(self, c_param = 1.4, use_rave = False, use_grave = False, use_puct = False, beta_const = 300, c_puct = 1.0, threshold = 30):
        # returns the best child based on UCT/RAVE/other_method is enabled

        def get_rave_ancestor(node):
            # for GRAVE: climb up to find a node with decent visits
            while node is not None:
                if node.visits > threshold:
                    return node
                node = node.parent
            return self  # fallback to current node

        def score(child):
            # base q value
            q_value = child.value / (child.visits + 1e-4)

            if use_puct:
                # puct score includes prior and visit count
                return q_value + c_puct * child.prior * math.sqrt(self.visits + 1) / (1 + child.visits)

            # regular UCT formula
            uct = c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4))

            if use_rave or use_grave:
                ancestor = get_rave_ancestor(self) if use_grave else self
                rave_n = ancestor.rave_visits[child.move]
                rave_q = ancestor.rave_value[child.move] / (rave_n + 1e-4) if rave_n > 0 else 0

                beta = rave_n / (rave_n + child.visits + 1e-4 + beta_const * rave_n * child.visits)
                value = (1 - beta) * q_value + beta * rave_q
            else:
                value = q_value

            return value + uct

        return max(self.children, key = score)

    def expand(self, use_puct):
        # expands one of the untried moves
        game = self.get_game_copy()
        if not self.untried_moves:
            self.untried_moves = game.valid_moves(game.current_player)

        move = self.untried_moves.pop()
        game.perform_move(move, game.current_player)
        new_state = game.from_board_to_state()
        child = MCTSNode(new_state, parent = self, move = move, player = game.current_player)

        # for PUCT: evaluate all moves to set priors
        if use_puct:
            move_scores = {}
            total_score = 0.0
            for m in game.valid_moves(game.current_player):
                temp_game = game.get_game_copy()
                temp_game.perform_move(m, temp_game.current_player)
                score = evaluate(temp_game, game.current_player)
                move_scores[m] = score
                total_score += score

            # normalize and assign prior to child
            if total_score > 0:
                child.prior = move_scores.get(move, 0) / total_score
            else:
                child.prior = 1 / len(game.valid_moves(game.current_player))

        self.children.append(child)
        return child

    def backpropagate(self, reward, visited_moves, use_rave = False):
        # updates the node and its parents with the reward
        self.visits += 1
        self.value += reward

        if use_rave:
            for move in visited_moves:
                self.rave_visits[move] += 1
                self.rave_value[move] += reward

        if self.parent:
            # if the parent node's player is the same, the reward remains the same.
            # if the parent node's player is different, negate the reward.
            if self.parent.player == self.player:
                self.parent.backpropagate(reward, visited_moves, use_rave)
            else:
                self.parent.backpropagate(-reward, visited_moves, use_rave)

    def rollout(self):
        # simulates a game by picking random moves
        game = self.get_game_copy()
        visited_moves = []
        while not game.is_terminal():
            move = random.choice(game.valid_moves(game.current_player))
            visited_moves.append(move)
            game.perform_move(move, game.current_player)

        result = game.result()
        reward = 1 if result > 0 else -1 if result < 0 else 0
        return reward, visited_moves

# === MCTS and Variants ===
# Base MCTS: UCT = Q + c * sqrt(log(N) / n)
# RAVE/GRAVE: blends AMAF with Q-values using beta
# PUCT: adds priors to UCT using PUCT formula
# SHOT: filters moves in halving rounds using fixed budget
# SHUSS: uses SHUSS score = Q + c * AMAF / p
class KalahMCTS:
    def __init__(self, game, iterations = 1000, use_rave = False, use_grave = False, use_puct = False, use_shot = False, use_shuss = False, c_param = 1.4, beta_const = 300, c_puct = 1.0, shuss_c = 128.0, threshold = 30):
        # stores all configs for the mcts algorithm
        self.iterations = iterations
        self.use_rave = use_rave                            # enable RAVE
        self.use_grave = use_grave                          # enable GRAVE
        self.use_puct = use_puct                            # enable PUCT
        self.use_shot = use_shot                            # enable SHOT
        self.use_shuss = use_shuss                          # enable SHUSS
        self.threshold = threshold

        self.c_param = c_param
        self.beta_const = beta_const
        self.c_puct = c_puct
        self.shuss_c = shuss_c

        self.root = MCTSNode(game.from_board_to_state(), player = game.current_player)

    def run_shot(self, root_node, budget):
        # SHOT: simulates all moves and prunes the worst half repeatedly
        game = root_node.get_game_copy()
        valid_moves = game.valid_moves(game.current_player)
        move_nodes = []

        # create child nodes for each move
        for move in valid_moves:
            g_copy = root_node.get_game_copy()
            g_copy.perform_move(move, g_copy.current_player)
            new_state = g_copy.from_board_to_state()
            child = MCTSNode(new_state, parent = root_node, move = move, player = g_copy.current_player)
            move_nodes.append((move, child))

        k = len(move_nodes)
        if k == 1:
            return move_nodes[0][0]

        total_k = k  # save initial move count for fixed log2 denominator
        while k > 1:
            simulations_per_node = budget // (k * int(math.log2(total_k + 1)))

            for _, node in move_nodes:
                for _ in range(simulations_per_node):
                    leaf = self.tree_policy(node)
                    reward, visited_moves = leaf.rollout()
                    leaf.backpropagate(reward, visited_moves)

            # sort by average value and halve
            move_nodes.sort(key = lambda x: x[1].value / (x[1].visits + 1e-4), reverse = True)
            k = k // 2
            move_nodes = move_nodes[:max(1, k)]

        return move_nodes[0][0]

    def run_shuss(self, root_node, budget, c = None):
        # SHUSS is like shot but uses a different scoring formula
        game = root_node.get_game_copy()
        valid_moves = game.valid_moves(game.current_player)
        move_nodes = []

        # create child nodes for each legal move
        for move in valid_moves:
            g_copy = root_node.get_game_copy()
            g_copy.perform_move(move, g_copy.current_player)
            new_state = g_copy.from_board_to_state()
            child = MCTSNode(new_state, parent = root_node, move = move, player = g_copy.current_player)
            move_nodes.append((move, child))

        k = len(move_nodes)
        total_k = k  # fixed for budget division
        if k == 1:
            return move_nodes[0][0]

        b = budget
        while k > 1:
            sims_per_node = b // (k * int(math.log2(total_k + 1)))

            for _, node in move_nodes:
                for _ in range(sims_per_node):
                    leaf = self.tree_policy(node)
                    reward, visited_moves = leaf.rollout()
                    leaf.backpropagate(reward, visited_moves, use_rave = True)

            # SHUSS scoring: μ + c * AMAF / p
            def shuss_score(node):
                q = node.value / (node.visits + 1e-4)
                amaf = root_node.rave_value[node.move]
                p = root_node.rave_visits[node.move] + 1e-4
                return q + (c or self.shuss_c) * (amaf / p)

            move_nodes.sort(key = lambda x: shuss_score(x[1]), reverse = True)
            k = k // 2
            move_nodes = move_nodes[:max(1, k)]

        return move_nodes[0][0]
    
    def best_move(self, game):
        # runs MCTS and returns the best move
        self.root = MCTSNode(game.from_board_to_state(), player = game.current_player)

        if self.use_shot:
            return self.run_shot(self.root, self.iterations)
        if self.use_shuss:
            return self.run_shuss(self.root, self.iterations, c = self.shuss_c)

        for _ in range(self.iterations):
            node = self.tree_policy(self.root)
            reward, visited_moves = node.rollout()
            node.backpropagate(reward, visited_moves, use_rave = (self.use_rave or self.use_grave))

        return self.root.best_child(
                c_param = self.c_param,
                use_rave = self.use_rave,
                use_grave = self.use_grave,
                use_puct = self.use_puct,
                beta_const = self.beta_const,
                c_puct = self.c_puct,
                threshold = self.threshold
            ).move

    def tree_policy(self, node):
        # goes down the tree picking best children or expanding new ones
        while not self.terminal_state(node):
            if not node.is_fully_expanded():
                return node.expand(self.use_puct)
            else:
                node = node.best_child(
                    c_param = self.c_param,
                    use_rave = self.use_rave,
                    use_grave = self.use_grave,
                    use_puct = self.use_puct,
                    beta_const = self.beta_const,
                    c_puct = self.c_puct
                )
        return node

    def terminal_state(self, node):
        # checks if the game is over from the node’s state
        game = node.get_game_copy()
        return game.is_terminal()