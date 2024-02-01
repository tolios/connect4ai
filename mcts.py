from connect4 import get_next_open_row, drop_piece, get_available_moves, winning_move
from math import sqrt, log
import random
from tqdm import tqdm

class MCTS:

    def __init__(self, pl, c = 2, samples = 1000, n_rollout = 1):
        self.pl = pl
        self.c = c
        self.samples = samples
        self.n_rollout = n_rollout
        self.tree = dict()
        self.nodes = dict()
        self.end_nodes = dict()
        self.n = 0

    def move(self, board):
        # set up root info
        # n, w
        self.nodes[0] = (0, 0)
        for _ in range(self.samples):
            sim_board = board.copy()
            leaf, trace, pl = self.select(sim_board)
            # check if leaf is an end node...
            if not (leaf in self.end_nodes):
                # trace doesn't contain leaf! (yet)
                # expand leaf if needed
                n, _ = self.nodes[leaf]
                if (n != 0):
                    # expand, therefore new leaf
                    trace.append(leaf)
                    leaf = self.expand(sim_board, leaf, pl)
                    # check if end node...
                    if not get_available_moves(sim_board):
                        # no moves...
                        if winning_move(sim_board, pl):
                            v_tot = 1 if pl == 1 else -1
                            n_tot = 1
                        else:
                            v_tot = 0
                            n_tot = 1
                        self.end_nodes[leaf] = v_tot
                    else:

                        # flip player
                        pl = 2 if pl == 1 else 1

                        # perform rollout(s)
                        v_tot = 0
                        n_tot = self.n_rollout
                        for _ in range(n_tot):
                            v_tot += simulate(sim_board, pl)
                        v_tot = v_tot/n_tot
                else:
                    # perform rollout(s)
                    v_tot = 0
                    n_tot = self.n_rollout
                    for _ in range(n_tot):
                        v_tot += simulate(sim_board, pl)
                    v_tot = v_tot/n_tot
            else:
                n_tot = 1
                v_tot = self.end_nodes[leaf]

            self.backprop(n_tot, v_tot, leaf, trace)
        # pick best move
        move = self.best_move(self.pl)
        # delete all info ...
        self.empty()
        return move
    
    def select(self, sim_board):
        # start player (bot) and root node
        pl = self.pl
        node = 0
        trace = []
        # goes down the tree to find best root.
        while node in self.tree:
            trace.append(node)
            N = self.nodes[node][0]
            best_child = None
            best_move = None
            best_ucb = -10000 # small
            for child, move in self.tree[node]:
                n, w = self.nodes[child]
                #calculate ucb (for appropriate player)
                ucb = UCB1(n, w, N, self.c, p = -1 if pl == 2 else 1)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
                    best_move = move
            # go down a node!
            drop_piece(sim_board, get_next_open_row(sim_board, best_move), best_move, pl)
            node = best_child
            # change player
            pl = 2 if pl == 1 else 1  

        # returns leaf, trace and the one that is to play
        return node, trace, pl
    
    def expand(self, sim_board, leaf, pl):
        # here expanding, it will fill the self.nodes, self.tree
        self.tree[leaf] = []
        moves = get_available_moves(sim_board)
        for move in moves:
            # generate new id
            self.n += 1
            self.tree[leaf].append((self.n, move))
            self.nodes[self.n] = (0, 0)
        # in the end pick one child as leaf
        leaf, move = random.choice(self.tree[leaf])
        # and act it...
        drop_piece(sim_board, get_next_open_row(sim_board, move), move, pl)
        # new leaf
        return leaf

    def backprop(self, n_new, w_new, leaf, trace):
        # only here the actual leaf will join
        trace.append(leaf)
        for node in trace:
            n, w = self.nodes[node]
            n += n_new
            w += w_new
            self.nodes[node] = (n, w)

    def best_move(self, pl):
        # starting from root, pick action with best value (for pl 2 is neg)
        best = -10000
        best_move = None
        s = 1 if pl == 1 else -1
        for child, move in self.tree[0]:
            n, w = self.nodes[child]

            if n != 0:
                pick = (s)*(w/n)
            else:
                pick = 0

            if pick > best:
                best = pick
                best_move = move
        return best_move
    
    def empty(self):
        self.n = 0
        self.tree = dict()
        self.nodes = dict()
        self.end_nodes = dict()

def UCB1(n, w, N, c, p = 1):
    if n != 0:
        return p*(w/n) + c*sqrt(log(N)/n)
    else:
        return 1000

def simulate(board, pl):
    sim_board = board.copy()
    while True:
        # check if won by opponent
        if winning_move(sim_board, 1 if pl == 2 else 2):
            return 1 if pl == 2 else -1
        moves = get_available_moves(sim_board)
        if not moves:
            return 0
        move = random.choice(moves)
        drop_piece(sim_board, get_next_open_row(sim_board, move), move, pl)

        pl = 1 if pl == 2 else 2

# board = create_board()

# a = MCTS(samples=100000, n_rollout=1)

# # TODO FIX GAME ENDING (win draw loss)
# a.move(board)
# print()