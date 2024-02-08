# TreeStrap from https://www.davidsilver.uk/wp-content/uploads/2020/03/bootstrapping.pdf

from connect4 import get_next_open_row, drop_piece, get_available_moves, value_position, create_board, winning_move, is_valid_location, ROW_COUNT, COLUMN_COUNT
import torch
from torch.nn import Linear, ReLU, Tanh, Module, Conv2d
from torch.optim import Adam
import random
import numpy as np

def board_transform(board):
    # return torch.from_numpy((board - (3/2)*(board!=0))*2).flatten().float()
    return torch.from_numpy((board - (3/2)*(board!=0))*2).unsqueeze(0).float()

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

class ConvolutionalHeuristic(Module):
    def __init__(self, hidden_dim = 128):
        super(ConvolutionalHeuristic, self).__init__()
        self.hidden_dim = hidden_dim
        # Define the convolutional layers
        self.conv1 = Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        # Define the fully connected layers
        self.fc1 = Linear(hidden_dim * COLUMN_COUNT * ROW_COUNT, 1)
        # Define the activation function
        self.activation = Tanh()
    
    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.activation(self.conv1(x))
        # Flatten the output for fully connected layers
        x = x.view(-1, self.hidden_dim * COLUMN_COUNT * ROW_COUNT)
        
        # Forward pass through fully connected layers
        x = self.activation(self.fc1(x))
        
        return x

    def update_network(self, other, tau=0.01):
        # Soft update of target network parameters
        for target_param, primary_param in zip(self.parameters(), other.parameters()):
            target_param.data.copy_(tau * primary_param.data + (1.0 - tau) * target_param.data)

class HeuristicNN(Module):

    def __init__(self, hidden_dim = 100):
        super().__init__()
        self.conv = Conv2d(1, COLUMN_COUNT*ROW_COUNT)
        self.lin1 = Linear(COLUMN_COUNT*ROW_COUNT, hidden_dim)
        self.relu1 = ReLU()
        self.lin2 = Linear(hidden_dim, 1)
        self.tanh = Tanh()
    
    def forward(self, x):
        return self.tanh(self.lin2(self.relu1(self.lin1(x))))

class TreeStrap:

    def __init__(self, pl, depth=4, hidden_dim = 100, lr = 0.0001, weight_decay=0.00001, epsilon = 0.0, tau = 0.01, T=0.1, n_max = 4):
        self.pl = pl
        self.self_pl = 1
        self.tau = tau
        self.depth = depth
        self.Hnn = ConvolutionalHeuristic(hidden_dim=hidden_dim)
        self.target_Hnn = ConvolutionalHeuristic(hidden_dim=hidden_dim)
        self.target_Hnn.update_network(self.Hnn, tau=1.0)
        self.optim = Adam(self.Hnn.parameters(), lr = lr, weight_decay=weight_decay)
        self.epsilon = epsilon
        self.T = T
        self.n_max = n_max

    def move(self, board):
        # determines col to move given position!
        _, move, _ = minimax(board, self.depth, self.pl == 1, self.heuristic, 0)
        return move
    
    def value_move(self, board, pl):
        # determines col to move given position!
        value, move, loss = minimax(board, self.depth, pl == 1, self.heuristic, 0, train = True, epsilon=self.epsilon)
        return value, move, loss

    def evaluate(self, board, player):
        return minimax(board, self.depth, player==1, self.heuristic, 0)[0]
    
    def heuristic(self, board, train = False):
        if not train:
            with torch.no_grad():
                x = board_transform(board)
                x = self.target_Hnn(x)
                return x.numpy().item()
        else:
            x = board_transform(board)
            x = self.Hnn(x)
            return x
    
    def self_play(self, iters = 1000):

        running_avg_loss = 0
        for iter in range(iters):
            self.self_pl = 1
            init_moves = self.generate_initial_moves()
            board = create_board()
            for _ in range(init_moves):
                # make random moves to get an init position (Exploration)
                col = random.choice(get_available_moves(board))
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, self.self_pl)
                self.self_pl = 2 if self.self_pl == 1 else 1
            game_over = False
            # game starts...
            tot_loss = 0
            n = init_moves
            while not game_over:
                n += 1
                # agent play
                self.optim.zero_grad()
                value, col, loss = self.value_move(board, self.self_pl)
                (loss).backward()
                self.optim.step()
                tot_loss += loss.item()
                running_avg_loss += loss.item()

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, self.self_pl)
                    if winning_move(board, self.self_pl): 
                        game_over = True
                        self.optim.zero_grad()
                        value = torch.tensor(1 if self.self_pl == 1 else -1)
                        hvalue = self.heuristic(board, train=True)
                        loss = (value - hvalue)**2
                        (loss).backward()
                        self.optim.step()
                else:
                    raise
                
                if not game_over:
                    if not get_available_moves(board):
                        game_over = True
                        self.optim.zero_grad()
                        value = torch.tensor(0)
                        hvalue = self.heuristic(board, train=True)
                        loss = (value - hvalue)**2
                        (loss).backward()
                        self.optim.step()
                        tot_loss += loss.item()
                        running_avg_loss += loss.item()
                
                self.self_pl = 2 if self.self_pl == 1 else 1
            
            #update target...
            self.target_Hnn.update_network(self.Hnn, tau=self.tau)

            print(f"Game: {iter}, Tot loss: {tot_loss}, Run. Avg: {running_avg_loss/(iter+1)}, Rounds {n}, Starting {init_moves}")
    
    def generate_initial_moves(self):
        # Compute the scores for each number of initial moves
        scores = np.array([-(n * (1/self.T)) for n in range(self.n_max + 1)])
        # Compute the probabilities using the softmax function
        probs = np.exp(scores) / np.sum(np.exp(scores))
        # Randomly select the number of initial moves based on the probabilities
        num_moves = np.random.choice(np.arange(len(probs)), p=probs)
        
        return num_moves

def minimax(board, depth, max_player, heuristic, loss, train = False, epsilon = 0):
    pick_random = False
    pl = 1 if max_player else 2

    sim_board = board.copy()
    
    # checks if game is finished
    value, finished = value_position(sim_board)

    if finished:
        loss += (heuristic(sim_board, train=train) - torch.tensor(value, dtype = float))**2
        return value, None, loss
    
    #if depth = 0, close
    if depth == 0 and heuristic:
        return heuristic(sim_board), None, loss

    
    value = MIN if max_player else MAX
    action = None
    
    moves = get_available_moves(sim_board)
    if random.random() < epsilon:
        pick_random = True
        random_i = random.randint(0, len(moves))

    for i, move in enumerate(moves):
        sim_board = board.copy()
        drop_piece(sim_board, get_next_open_row(sim_board, move), move, pl)

        v, _, loss = minimax(sim_board, depth - 1,not max_player, heuristic, loss, train=train, epsilon=epsilon)

        if max_player:
            if v > value:
                value = v
                action = move
        else:
            if v < value:
                value = v
                action = move
        
        if pick_random:
            if i == random_i:
                # we pick a random action and therefore value
                action = move
                value = v
                break

    # having found the value of the upstream node, it adds the losses of its prediction!
    loss += (heuristic(board, train=train) - torch.tensor(value, dtype = float))**2

    return value, action, loss


#TODO - Add some sort of epsilon greedy selection in training!