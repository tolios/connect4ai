from connect4 import get_next_open_row, drop_piece, get_available_moves, winning_move
from math import sqrt, log
import random

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
        return self.mcts(board, self.pl)[0]

    def evaluate(self, board, player):
        return self.mcts(board, player)[1]

    def mcts(self, board, start_pl):
        if winning_move(board, 1):
            return None, 1
        elif winning_move(board, 2):
            return None, -1
        else:
            if not get_available_moves(board):
                return None, 0
        
        # set up root info
        # n, w
        self.nodes[0] = (0, 0)
        for _ in range(self.samples):
            sim_board = board.copy()
            # returns leaf, trace and starting pl to play
            leaf, trace, pl = self.select(sim_board, start_pl)
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
                    if winning_move(sim_board, pl):
                        v_tot = 1 if pl == 1 else -1
                        n_tot = 1
                        self.end_nodes[leaf] = v_tot
                    elif not get_available_moves(sim_board):
                        v_tot = 0
                        n_tot = 1
                        self.end_nodes[leaf] = v_tot
                    else:
                        # typical rollout
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
                    # but first check if opp won
                    if winning_move(sim_board, 1 if pl == 2 else 2):
                        v_tot = -1 if pl == 1 else 1
                        n_tot = 1
                        self.end_nodes[leaf] = v_tot
                    elif not get_available_moves(sim_board):
                        v_tot = 0
                        n_tot = 1
                        self.end_nodes[leaf] = v_tot
                    else:
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
        move = self.best_move(start_pl)
        # get value...
        n, w = self.nodes[0]
        # delete all info ...
        self.empty()
        return move, w/n
    
    def select(self, sim_board, pl):
        # start player
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
        for i, node in enumerate(trace[::-1]):
            n, w = self.nodes[node]
            n += n_new
            w += w_new * (0.99**(i))
            self.nodes[node] = (n, w)

    def best_move(self, pl):
        # starting from root, pick action with most visits!
        best = -10000
        best_move = None
        for child, move in self.tree[0]:
            n = self.nodes[child][0]
            if n > best:
                best = n
                best_move = move
        return best_move
    
    def empty(self):
        self.n = 0
        self.tree = dict()
        self.nodes = dict()
        self.end_nodes = dict()

def UCB1(n, w, N, c, p = 1):
    if n != 0:
        return (p*(w/n)) + c*sqrt(log(N)/n)
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
