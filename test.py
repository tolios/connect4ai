from connect4 import create_board, get_available_moves, get_next_open_row, print_board, drop_piece, winning_move, value_position
import random
import numpy as np
import itertools

class GenMoves:
    def __init__(self, pl, gens = 10, N = 50, topk = 10, p_mut = 0.1, p_depth = 0.1):
        self.pl = pl
        self.gens = gens
        self.N = N
        self.topk = topk
        self.p_mut = p_mut
        self.p_depth = p_depth

    def move(self, board):
        topk = generate_nash(board, self.pl, gens=self.gens, N=self.N, topk=self.topk, p_mut=self.p_mut, p_depth=self.p_depth)
        return topk[0].move()
    
    def evaluate(self, board, player):
        topk = generate_nash(board, self.pl, gens=self.gens, N=self.N, topk=self.topk, p_mut=self.p_mut, p_depth=self.p_depth)
        return sum(linet.eval for linet in topk)/len(topk)

def generate_random_couples(topk, num_children):
    all_couples = list(itertools.combinations(range(topk), 2))
    random_couples = random.sample(all_couples, num_children)
    return random_couples

class line:

    def __init__(self, moves, eval):
        self.moves = moves
        self.eval = eval
        self.sensibility = np.array([0., 0.])

    def move(self):
        return self.moves[0]

    def visualize(self, board, player):
        sim_board = board.copy()
        for move in self.moves:
            drop_piece(sim_board, get_next_open_row(sim_board, move), move, player)
            player = 2 if player == 1 else 1
        
        return sim_board

    def aggr(self):
        return np.sum(self.sensibility)
    
    def mutate(self, p_mut = 0.1, p_depth = 0.1):
        # takes line as input and changes with a probability p_mut
        # it goes up the depth with probability p_depth for each increase.
        # Check if mutation occurs
        if random.random() < p_mut:
            depth = 1
            # Determine the depth to go back
            while random.random() < p_depth:
                depth += 1
            # Go back to the chosen depth unless max_depth
            if depth >= len(self.moves):
                moves = []
            else:
                moves = self.moves[:-depth]
            # board to simulate
            sim_board = create_board()
            player = 1
            finished = False
            line_moves = []

            while not finished:
                if moves:
                    move = moves.pop(0)
                else:
                    move = random.choice(get_available_moves(sim_board))
                line_moves.append(move)
                drop_piece(sim_board, get_next_open_row(sim_board, move), move, player)
                val, finished = value_position(sim_board)

                player = 2 if player == 1 else 1
            
            
            self.moves = line_moves
            self.eval = val
            # since mutated, sensibility is unknown
            self.sensibility = np.array([0., 0.])
        
        return self

    def new(self):
        return line(self.moves, self.eval)
    
    def combine(self, other):
        # combine two lines to a new line with random playout at not common
        sim_board = create_board()
        player = 1
        line_moves = []
        for movel, mover in zip(self.moves, other.moves):
            if movel == mover:
                drop_piece(sim_board, get_next_open_row(sim_board, movel), movel, player)
                player = 2 if player == 1 else 1
                line_moves.append(movel)
            else:
                break
        # random  playout the rest...
        val, finished = value_position(sim_board)

        while not finished:
            move = random.choice(get_available_moves(sim_board))
            line_moves.append(move)
            drop_piece(sim_board, get_next_open_row(sim_board, move), move, player)
            val, finished = value_position(sim_board)

            player = 2 if player == 1 else 1

        return line(line_moves, val)

def start_pop(board, player, N):
    sim_board = board.copy()
    made = set()
    pop = list()
    while len(pop)<N:
        line_ = generate_line(sim_board, player)
        # if not (tuple(line_.moves) in made):
        #     made.add(tuple(line_.moves))
        pop.append(line_)

    return pop

def generate_line(board, player):
    sim_board = board.copy()
    line_moves = []

    val, finished = value_position(sim_board)

    while not finished:
        move = random.choice(get_available_moves(sim_board))
        line_moves.append(move)
        drop_piece(sim_board, get_next_open_row(sim_board, move), move, player)
        val, finished = value_position(sim_board)

        player = 2 if player == 1 else 1

    return line(line_moves, val)

#TODO make better compare. Seems biased with start player (most evals are for him)
def compare(line1, line2, player):
    dl1, dl2 = np.array([0., 0.]), np.array([0., 0.])

    if line1.moves != line2.moves:

        for move1, move2 in zip(line1.moves, line2.moves):
            if move1 != move2:
                break
    
            player = 2 if player == 1 else 1
        
        eval1, eval2 = line1.eval, line2.eval

        if player == 2:
            # flips signs to unify symbols
            eval1, eval2 = -eval1, -eval2

        # add depth decay, to prioritize faster wins

        match eval1 - eval2:
            case 0:
                dl1[player-1] = (0.9)**(len(line1.moves)-1)
                dl2[player-1] = (0.9)**(len(line2.moves)-1)
            case 1:
                dl1[player-1] = (0.9)**(len(line1.moves)-1)
            case 2:
                dl1[player-1] = (0.9)**(len(line1.moves)-1)
            case -1:
                dl2[player-1] = (0.9)**(len(line2.moves)-1)
            case -2:
                dl2[player-1] = (0.9)**(len(line2.moves)-1) 
    else:
        # if the same, then grow the sensibility of start player for both moves (tied)
        dl1[player-1] = (0.9)**(len(line1.moves)-1)
        dl2[player-1] = (0.9)**(len(line2.moves)-1)
    
    # updates sensibilities...
    line1.sensibility += dl1
    line2.sensibility += dl2

def generate_nash(board, player, gens = 10, N = 200, topk = 50, p_mut = 0.1, p_depth = 0.1):
    if topk > N or topk % 2 != 0:
        raise

    sim_board = board.copy()
    population = start_pop(sim_board, player, N)

    for gen in range(1, gens + 1):
        # 0s out all sensibilities for recalc
        population = [linei.new() for linei in population]
        # compare all to get best
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                line_i, line_j = population[i], population[j]
                compare(line_i, line_j, player)
        
        # rank
        population = sorted(population, key=lambda obj: obj.aggr(), reverse=True)

        # get topk
        topk_pop = population[:topk]
        population = topk_pop
        print(f"Gen {gen} avg topk Sensibility/N", sum(pop.aggr() for pop in topk_pop)/N)

        # procreation...
        couples = generate_random_couples(topk, N-topk)
        for l, r in couples:
            linel, liner = topk_pop[l], topk_pop[r]
            population.append(linel.combine(liner).mutate(p_mut = p_mut, p_depth = p_depth))

    # trust only topk...
    return population[:topk]

if __name__ == "__main__":

    board = create_board()
    print_board(board)

    player = 1

    drop_piece(board, get_next_open_row(board, 1), 1, player)
    player = 2 if player == 1 else 1
    drop_piece(board, get_next_open_row(board, 1), 1, player)
    player = 2 if player == 1 else 1
    drop_piece(board, get_next_open_row(board, 2), 2, player)
    player = 2 if player == 1 else 1
    drop_piece(board, get_next_open_row(board, 2), 2, player)
    player = 2 if player == 1 else 1
    drop_piece(board, get_next_open_row(board, 0), 0, player)
    player = 2 if player == 1 else 1
    # drop_piece(board, get_next_open_row(board, 3), 3, player)
    # player = 2 if player == 1 else 1

    print(player)
    print_board(board)

    topk = generate_nash(board, player, gens=10, topk=10, N=50, p_mut=0.2, p_depth=0.5)

    print_board(topk[0].visualize(board, player))
    print(sum(linet.eval for linet in topk)/len(topk))
    print(topk[0].move())