from connect4 import get_next_open_row, drop_piece, get_available_moves, value_position
from scipy.signal import convolve2d
import numpy as np

MAX = 1000
MIN = -1000

class TranspositionTable:
    def __init__(self):
        self.table = dict()

    def store(self, state, value):
        key = str(state)
        self.table[key] = value

    def lookup(self, state):
        key = str(state)
        return self.table.get(key, False)
    
    def clear(self):
        # empty transposition table for next round...
        self.table = dict()

# if use of depth, coupled with heuristics
    
horizontal_kernel = np.array([[ 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(3, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

class MiniMax:

    def __init__(self, depth=4) -> None:
        self.transpositionTable = TranspositionTable()
        self.depth = depth

    def move(self, board):
        # determines col to move given position!
        _, move = minimax(board, self.depth, False, MIN, MAX, self.transpositionTable, heuristic=self.heuristic) # False since we always pick bot for minimizer
        self.transpositionTable.clear()
        return move

    def evaluate(self, board, player):
        return minimax(board, self.depth, player==1, MIN, MAX, self.transpositionTable, heuristic=self.heuristic)[0]
    
    def heuristic(self, board):
        activations = 0
        for kernel in detection_kernels:
            activations += (convolve2d(board == 1, kernel, mode="valid") == 3).sum()
            activations -= (convolve2d(board == 2, kernel, mode="valid") == 3).sum()
        
        value = np.tanh(activations)

        if value > 0:
            value += -0.01
        else:
            value += +0.01
        
        return np.tanh(activations)
    
def minimax(board, depth, max_player, alpha, beta, transpositionTable, heuristic = None):

    pl = 1 if max_player else 2

    sim_board = board.copy()
    
    # checks if game is finished
    value, finished = value_position(sim_board)

    if finished:
        return value, None
    
    #if depth = 0, close
    if depth == 0 and heuristic:
        return heuristic(sim_board), None

    
    value = MIN if max_player else MAX
    action = None
    
    moves = get_available_moves(sim_board)

    for move in moves:
        sim_board = board.copy()
        drop_piece(sim_board, get_next_open_row(sim_board, move), move, pl)

        # check if it exists in transposition table...
        v = transpositionTable.lookup(sim_board)

        if not v:
            v, _ = minimax(sim_board, depth - 1,not max_player, alpha, beta, transpositionTable, heuristic=heuristic)

        if max_player:
            if v > value:
                value = v
                action = move
            alpha = max(alpha, v)
            # Alpha Beta Pruning 
            if beta <= alpha: 
                break
            # store values!
            transpositionTable.store(sim_board, v)
        else:
            if v < value:
                value = v
                action = move
            beta = min(beta, v)
            # Alpha Beta Pruning 
            if beta <= alpha: 
                break
            # store values!
            transpositionTable.store(sim_board, v)

    return value, action
