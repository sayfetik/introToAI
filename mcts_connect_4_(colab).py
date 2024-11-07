"""Original file is located at
    https://colab.research.google.com/drive/1l8uGFs4e13fUZXZeo2Rrq1Wi6GiiPKiX

# There are four phases of implementing MCTS:

- Selection
- Expansion
- Rollout
- Backpropagation

# Meta
### To do: Assign the exploration constant
"""

import math

class GameMeta:
    PLAYERS = {'none': 0, 'one': 1, 'two': 2}
    OUTCOMES = {'none': 0, 'one': 1, 'two': 2, 'draw': 3}
    INF = float('inf')
    ROWS = 6
    COLS = 7


class MCTSMeta:
    # Exploration constant
    EXPLORATION = math.sqrt(2)

"""# Game class (Connect 4)"""

from copy import deepcopy
import numpy as np


class ConnectState:
    def __init__(self):
        self.board = [[0] * GameMeta.COLS for _ in range(GameMeta.ROWS)]
        self.to_play = GameMeta.PLAYERS['one']
        self.height = [GameMeta.ROWS - 1] * GameMeta.COLS
        self.last_played = []

    def get_board(self):
        return deepcopy(self.board)

    def move(self, col):
        self.board[self.height[col]][col] = self.to_play
        self.last_played = [self.height[col], col]
        self.height[col] -= 1
        self.to_play = GameMeta.PLAYERS['two'] if self.to_play == GameMeta.PLAYERS['one'] else GameMeta.PLAYERS['one']

    def get_legal_moves(self):
        return [col for col in range(GameMeta.COLS) if self.board[0][col] == 0]

    def check_win(self):
        if len(self.last_played) > 0 and self.check_win_from(self.last_played[0], self.last_played[1]):
            return self.board[self.last_played[0]][self.last_played[1]]
        return 0

    def check_win_from(self, row, col):
        player = self.board[row][col]
        """
        Last played action is at (row, col)
        Check surrounding 7x7 grid for a win
        """

        consecutive = 1
        # Check horizontal
        tmprow = row
        while tmprow + 1 < GameMeta.ROWS and self.board[tmprow + 1][col] == player:
            consecutive += 1
            tmprow += 1
        tmprow = row
        while tmprow - 1 >= 0 and self.board[tmprow - 1][col] == player:
            consecutive += 1
            tmprow -= 1

        if consecutive >= 4:
            return True

        # Check vertical
        consecutive = 1
        tmpcol = col
        while tmpcol + 1 < GameMeta.COLS and self.board[row][tmpcol + 1] == player:
            consecutive += 1
            tmpcol += 1
        tmpcol = col
        while tmpcol - 1 >= 0 and self.board[row][tmpcol - 1] == player:
            consecutive += 1
            tmpcol -= 1

        if consecutive >= 4:
            return True

        # Check diagonal
        consecutive = 1
        tmprow = row
        tmpcol = col
        while tmprow + 1 < GameMeta.ROWS and tmpcol + 1 < GameMeta.COLS and self.board[tmprow + 1][tmpcol + 1] == player:
            consecutive += 1
            tmprow += 1
            tmpcol += 1
        tmprow = row
        tmpcol = col
        while tmprow - 1 >= 0 and tmpcol - 1 >= 0 and self.board[tmprow - 1][tmpcol - 1] == player:
            consecutive += 1
            tmprow -= 1
            tmpcol -= 1

        if consecutive >= 4:
            return True

        # Check anti-diagonal
        consecutive = 1
        tmprow = row
        tmpcol = col
        while tmprow + 1 < GameMeta.ROWS and tmpcol - 1 >= 0 and self.board[tmprow + 1][tmpcol - 1] == player:
            consecutive += 1
            tmprow += 1
            tmpcol -= 1
        tmprow = row
        tmpcol = col
        while tmprow - 1 >= 0 and tmpcol + 1 < GameMeta.COLS and self.board[tmprow - 1][tmpcol + 1] == player:
            consecutive += 1
            tmprow -= 1
            tmpcol += 1

        if consecutive >= 4:
            return True

        return False

    def game_over(self):
        return self.check_win() or len(self.get_legal_moves()) == 0

    def get_outcome(self):
        if len(self.get_legal_moves()) == 0 and self.check_win() == 0:
            return GameMeta.OUTCOMES['draw']

        return GameMeta.OUTCOMES['one'] if self.check_win() == GameMeta.PLAYERS['one'] else GameMeta.OUTCOMES['two']

    def print(self):
        print('=============================')

        for row in range(GameMeta.ROWS):
            for col in range(GameMeta.COLS):
                print('| {} '.format('X' if self.board[row][col] == 1 else 'O' if self.board[row][col] == 2 else ' '), end='')
            print('|')

        print('=============================')

"""# Selection phase
### To do: Calculate the Exploration term
"""

import math

class Node:
    def __init__(self, move, parent):
        # Initialize a Node object with a specific move and its parent node.
        self.move = move
        self.parent = parent
        self.N = 0  # Number of visits to this node.
        self.Q = 0  # Total reward obtained from this node.
        self.children = {}  # Dictionary to store child nodes with their moves as keys.
        self.outcome = GameMeta.PLAYERS['none']  # Current game outcome, initialized to 'none'.

    def add_children(self, children: dict) -> None:
        """
        Add child nodes to the current node.

        Args:
            children (dict): Dictionary containing child nodes with their moves as keys.
        """
        for child in children:
            self.children[child.move] = child

    def value(self, explore: float = MCTSMeta.EXPLORATION):
        """
        Calculate the value of the node, balancing exploration and exploitation.

        Args:
            explore (float): Exploration parameter controlling the balance between exploration and exploitation.

        Returns:
            float: Value of the node.
        """
        if self.N == 0:
            # If the node has not been visited, prioritize it for exploration.
            return 0 if explore == 0 else GameMeta.INF
        else:
            # Calculate the value using UCB1 formula (Upper Confidence Bound 1).
            exploration_term = explore * math.sqrt(sum(child.N for child in self.children.values())) / (self.N + sum(child.N for child in self.children.values()))
            return self.Q / self.N + exploration_term

import random
import time
from copy import deepcopy

class MCTS:
    def __init__(self, state=ConnectState()):
        """
        Initialize the MCTS object with the given initial game state.

        Args:
            state (ConnectState): Initial game state (default is an empty Connect 4 state).
        """
        self.root_state = deepcopy(state)  # Deep copy of the initial game state.
        self.root = Node(None, None)  # Create the root node for the search tree.
        self.run_time = 0  # Variable to store the total runtime of the MCTS algorithm.
        self.node_count = 0  # Variable to count the total number of nodes created during the search.
        self.num_rollouts = 0  # Variable to count the total number of rollouts performed during the search.

    def select_node(self) -> tuple:
        """
        Select a node in the search tree for exploration using the MCTS algorithm.

        Returns:
            tuple: Selected node and the corresponding game state.
        """
        node = self.root  # Start the selection process from the root node.
        state = deepcopy(self.root_state)  # Create a deep copy of the current game state for simulation.

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]

            node = random.choice(max_nodes)  # Randomly choose a child node based on UCT values.
            state.move(node.move)  # Apply the corresponding move to the game state.

            if node.N == 0:
                # If the node has not been visited, return it for expansion.
                return node, state

        if self.expand(node, state):
            # If the node is not a terminal state, expand it and select a child node randomly.
            node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state


    def move(self, move):
        """
        Update the MCTS tree and game state based on the chosen move.

        Args:
            move (int): Move to be played.
        """
        if move in self.root.children:
            self.root_state.move(move)  # Apply the chosen move to the game state.
            self.root = self.root.children[move]  # Update the root node to the corresponding child node.
            return

        self.root_state.move(move)  # Apply the chosen move to the game state.
        self.root = Node(None, None)  # Reset the root node to a new node corresponding to the updated game state.

    def statistics(self) -> tuple:
        """
        Get the statistics of the MCTS search.

        Returns:
            tuple: Number of rollouts performed and total runtime of the search.
        """
        return self.num_rollouts, self.run_time  #

"""# Expansion phase"""

def expand(self, parent: Node, state: ConnectState) -> bool:
        """
        Expand the given node by adding child nodes corresponding to legal moves.

        Args:
            parent (Node): Parent node to be expanded.
            state (ConnectState): Current game state corresponding to the parent node.

        Returns:
            bool: True if the node is expanded, False if it's a terminal state.
        """
        if state.game_over():
            return False  # If the game is over, do not expand further.

        children = [Node(move, parent) for move in state.get_legal_moves()]  # Create child nodes for legal moves.
        parent.add_children(children)  # Add the child nodes to the parent node.
        return True  # Node is successfully expanded.

MCTS.expand = expand

"""# Rollout/Simulation Phase"""

def roll_out(self, state: ConnectState) -> int:
        """
        Perform a random rollout from the given game state until a terminal state is reached.

        Args:
            state (ConnectState): Current game state for the rollout.

        Returns:
            int: Outcome of the rollout (-1 for player 1 win, 1 for player 2 win, 0 for draw).
        """
        while not state.game_over():
            state.move(random.choice(state.get_legal_moves()))  # Choose a random move and apply it to the game state.

        return state.get_outcome()  # Return the outcome of the game.
MCTS.roll_out = roll_out

"""# Backpropagation Phase
### To do: Update the values
"""

def back_propagate(self, node: Node, turn: int, outcome: int) -> None:
        """
        Backpropagate the outcome of a rollout up the tree, updating node statistics.

        Args:
            node (Node): Node to start the backpropagation.
            turn (int): Player to whom the outcome is favorable (-1 for player 1, 1 for player 2).
            outcome (int): Outcome of the game (-1, 0, or 1).
        """
        reward = 0 if outcome == turn else 1  # Calculate reward based on the current player's perspective.

        while node is not None:
            node.N += 1  # Increment the visit count of the node.
            node.Q += reward  # Update the total reward of the node.
            node = node.parent  # Move to the parent node for further backpropagation.
            if outcome == GameMeta.OUTCOMES['draw']:
                reward = 0  # If the game is a draw, set reward to 0.
            else:
                reward = 1 - reward  # Switch reward for the opponent player.
MCTS.back_propagate = back_propagate

"""# Combining the Four Phases
### To do: find outcome from rollout
"""

def search(self, time_limit: int):
        """
        Perform the MCTS search for a specified time limit.

        Args:
            time_limit (int): Time limit for the MCTS search in seconds.
        """
        start_time = time.process_time()  # Record the start time of the search.

        num_rollouts = 0  # Initialize the number of rollouts performed.
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()  # Select a node for exploration.
            outcome =self.roll_out(state)  # Perform a rollout from the selected node.
            self.back_propagate(node, state.to_play, outcome)  # Backpropagate the rollout outcome.
            num_rollouts += 1  # Increment the rollout count.

        run_time = time.process_time() - start_time  # Calculate the total runtime of the search.
        self.run_time = run_time  # Update the run_time variable.
        self.num_rollouts = num_rollouts  # Update the num_rollouts variable.

MCTS.search = search

"""# Choosing Best Action"""

def best_move(self):
        """
        Determine the best move to play based on the MCTS search results.

        Returns:
            int: Best move to play.
        """
        if self.root_state.game_over():
            return -1  # If the game is over, no valid move can be made.

        max_value = max(self.root.children.values(), key=lambda n: n.N).N  # Find the maximum visit count among child nodes.
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]  # Find nodes with the maximum visit count.
        best_child = random.choice(max_nodes)  # Randomly choose one of the nodes with the maximum visit count as the best move.

        return best_child.move  # Return the move corresponding to the best child node.

MCTS.best_move = best_move

"""# Playing with the agent"""

import time
def play():
    """
    Play the game with MCTS agent
    """

    state = ConnectState()
    mcts = MCTS(state)
    player_name = input("Please enter your name: ")

    while not state.game_over():
        print("Current state:")
        state.print()

        user_move = int(input("Enter a move: "))
        while user_move not in state.get_legal_moves():
            print("Illegal move")
            user_move = int(input("Enter a move: "))

        state.move(user_move)
        mcts.move(user_move)

        state.print()

        if state.game_over():
            print(f"Congratulations you won! {player_name}")
            break

        print("Thinking...")

        mcts.search(5)
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        move = mcts.best_move()

        print("MCTS chose move: ", move)

        state.move(move)
        mcts.move(move)

        if state.game_over():
            print(f"MCTS won! Better luck next time {player_name}")
            state.print()
            break


if __name__ == "__main__":
    play()
