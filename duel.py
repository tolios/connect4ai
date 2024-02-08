from connect4 import create_board, is_valid_location, get_next_open_row, drop_piece, winning_move, get_available_moves
from minimax import MiniMax
from mcts import MCTS
from rootstrap import RootStrap
from treestrap import TreeStrap
from tqdm import tqdm

def duel(agent1, agent2, rounds = 10):
    wins1 = 0
    wins2 = 0
    draw = 0

    if agent1.pl == agent2.pl:
        raise

    for _ in tqdm(range(rounds)):
        board = create_board()
        game_over = False

        while not game_over:
            # first agent
            col = agent1.move(board)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, agent1.pl)
                if winning_move(board, agent1.pl): 
                    game_over = True
                    wins1 += 1
            else:
                raise
            # second agent
            if not game_over:
                col = agent2.move(board)
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, agent2.pl)
                    if winning_move(board, agent2.pl): 
                        game_over = True
                        wins2 += 1
                else:
                    raise
            
            if not game_over:
                if not get_available_moves(board):
                    game_over = True
                    draw += 1
    
    return (wins1, wins2, draw)

if __name__ == "__main__":

    agent1 = MCTS(1, samples=1000, c = 2)
    # agent1 = MiniMax(1)
    agent2 = TreeStrap(2, hidden_dim=100, lr=0.0001, weight_decay=0.0000001)

    agent2.self_play(iters=100)

    print(duel(agent1, agent2, rounds=20))
