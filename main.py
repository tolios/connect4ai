from connect4 import create_board, print_board, draw_board, is_valid_location, get_next_open_row, drop_piece, winning_move, get_available_moves, COLUMN_COUNT, ROW_COUNT, BLACK, RED, YELLOW, GREEN, SQUARESIZE, RADIUS, width, screen
import pygame
import sys
import math
from minimax import MiniMax
import mlflow

agent = MiniMax(depth=7)
 
board = create_board()
print_board(board)
game_over = False

#initalize pygame
pygame.init()

draw_board(board)
pygame.display.update()
 
myfont = pygame.font.SysFont("monospace", 75)

while not game_over:
 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
 
        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            posx = event.pos[0]
            # if turn == 0:
            pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
            # else: 
            #     pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
        pygame.display.update()
 
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            #print(event.pos)
            # Ask for Player 1 Input
            posx = event.pos[0]
            col = int(math.floor(posx/SQUARESIZE))

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 1)

                if winning_move(board, 1):
                    label = myfont.render("Player 1 wins!!", 1, RED)
                    screen.blit(label, (40,10))
                    game_over = True
            else:
                raise

            print_board(board)
            draw_board(board)
 
            if not game_over:
                # # Ask for Player 2 Input              
                col = agent.move(board)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, 2)

                    if winning_move(board, 2):
                        label = myfont.render("Player 2 wins!!", 1, YELLOW)
                        screen.blit(label, (40,10))
                        game_over = True
                else:
                    raise
    
                print_board(board)
                draw_board(board)

                if not game_over:

                    if not get_available_moves(board):
                        label = myfont.render("Draw ...", 1, GREEN)
                        screen.blit(label, (40,10))
                        game_over = True
                        print_board(board)
                        draw_board(board)
 
            if game_over:
                pygame.time.wait(3000)
