from connect4 import create_board, print_board, draw_board, is_valid_location, get_next_open_row, drop_piece, winning_move, get_available_moves, BLACK, RED, YELLOW, GREEN, SQUARESIZE, RADIUS, width, size
import pygame
import sys
import math
from minimax import MiniMax
from mcts import MCTS

agent = MCTS(2, samples=2000, c = 1)

human_pl = 1 if agent.pl == 2 else 2
 
board = create_board()
print_board(board)
game_over = False

screen = pygame.display.set_mode(size)

#initalize pygame
pygame.init()

draw_board(board, screen)
pygame.display.update()
 
myfont = pygame.font.SysFont("monospace", 75)

while not game_over:
 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
 
        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            posx = event.pos[0]
            pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
        pygame.display.update()
 
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            posx = event.pos[0]
            col = int(math.floor(posx/SQUARESIZE))

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, human_pl)

                if winning_move(board, human_pl):
                    label = myfont.render(f"Player {human_pl} wins!!", 1, RED)
                    screen.blit(label, (40,10))
                    game_over = True
            else:
                raise

            print_board(board)
            draw_board(board, screen)
 
            if not game_over:
                # # Ask for Player 2 Input              
                col = agent.move(board)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, agent.pl)

                    if winning_move(board, agent.pl):
                        label = myfont.render(f"Player {agent.pl} wins!!", 1, YELLOW)
                        screen.blit(label, (40,10))
                        game_over = True
                else:
                    raise
    
                print_board(board)
                draw_board(board, screen)

                if not game_over:

                    if not get_available_moves(board):
                        label = myfont.render("Draw ...", 1, GREEN)
                        screen.blit(label, (40,10))
                        game_over = True
                        print_board(board)
                        draw_board(board, screen)
 
            if game_over:
                pygame.time.wait(3000)
