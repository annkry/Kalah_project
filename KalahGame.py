# Core game logic for Kalah including board setup, move execution, capturing, extra turns, and game-end checking.

class KalahGame:
    def __init__(self):
        # board indices:
        #  0-5   : Player 0’s pits (each starting with 4 stones)
        #  6     : Player 0’s store
        #  7-12  : Player 1’s pits (each starting with 4 stones)
        #  13    : Player 1’s store
        self.board = [4] * 6 + [0] + [4] * 6 + [0]
        self.current_player = 0  # 0 is player 0, 1 is player 1
        self.last_pit = None

    def from_board_to_state(self):
        # returns the current board and player as a state tuple
        return (tuple(self.board), self.current_player)

    def valid_moves(self, player):
        # returns list of non-empty pits that player can move from
        # player 0 uses pits 0-5, player 1 uses pits 7-12
        start = player * 7
        moves = [i for i in range(start, start + 6) if self.board[i] > 0]
        return moves or [None] # if no moves, return [None]

    def perform_move(self, move, player):
        if move is None:
            return
        index = move
        stones = self.board[index]
        self.board[index] = 0

        # drop stones one by one in the next pits
        while stones > 0:
            index = (index + 1) % 14
            # skip opponent's store:
            if index == (6 if player == 1 else 13):
                continue
            self.board[index] += 1
            stones -= 1

        # capture rule: if the last stone lands in an empty pit on the player's side
        # and the opposite pit contains stones, then capture both
        if 0 <= index < 6 and player == 0 and self.board[index] == 1 and self.board[12 - index] > 0:
            self.board[6] += self.board[index] + self.board[12 - index]
            self.board[index] = self.board[12 - index] = 0
        elif 7 <= index < 13 and player == 1 and self.board[index] == 1 and self.board[12 - index] > 0:
            self.board[13] += self.board[index] + self.board[12 - index]
            self.board[index] = self.board[12 - index] = 0

        # extra-turn rule: if the last stone lands in the player's store,
        # then the same player goes again. Otherwise, swap players
        self.last_pit = index
        if index != (6 if player == 0 else 13):
            self.current_player = 1 - player

    def is_on_player_side(self, ending_pit, player):
        # checks if a pit is on the player’s side
        return (0 <= ending_pit <= 6) if player == 0 else (7 <= ending_pit <= 13)
    
    def get_seeds(self, ending_pit):
        # returns number of seeds in a specific pit
        return self.board[ending_pit]
    
    def get_opposite_pit(self, ending_pit):
        # returns the opposite pit index for a given pit
        if ending_pit == 6 and ending_pit == 13:
            return None
        else:
            return 12 - ending_pit
        
    def get_store(self, player):
        # returns index of the player’s store
        return self.board[6] if player == 0 else self.board[13]
        
    def is_terminal(self):
       # checks if one side of the board is completely empty
        return sum(self.board[:6]) == 0 or sum(self.board[7:13]) == 0

    def result(self):
        # returns the score difference between player 0 and player 1
        # positive means player 0 wins, negative means player 1 wins
        return sum(self.board[:7]) - sum(self.board[7:14])
    
    def get_game_copy(self):
        # returns a deep copy of the current game state
        new_game = KalahGame()
        new_game.board = self.board[:]
        new_game.current_player = self.current_player
        new_game.last_pit = self.last_pit
        return new_game

    
    def print_board(self):
        """
            Visualize the board in a readable layout.
        
                 [12] [11] [10] [ 9] [ 8] [ 7]
         [13]                                     [ 6]
                 [ 0] [ 1] [ 2] [ 3] [ 4] [ 5]
        """
        print("\nCurrent Board state:")
        # top row: player 1’s pits
        print("    ", end="")
        for i in range(12, 6, -1):
            print(f"{self.board[i]:2d} ", end="")
        print()
        # stores: player 1’s on the left, player 0’s on the right
        print(f"{self.board[13]:2d} ", end="")
        print(" " * 20, end="")
        print(f"{self.board[6]:2d}")
        # bottom row: player 0’s pits
        print("    ", end="")
        for i in range(0, 6):
            print(f"{self.board[i]:2d} ", end="")
        print("\n")