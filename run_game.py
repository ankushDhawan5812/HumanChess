import sys
import pygame
import chess
from sv_move import return_next_move  # Your move function

# Pygame setup
pygame.init()
BOARD_SIZE = 480               # Chess board size
HISTORY_HEIGHT = 120          # Height for win% history plot
WINDOW_HEIGHT = BOARD_SIZE + HISTORY_HEIGHT
SQ_SIZE = BOARD_SIZE // 8     # Square size
FPS = 30

# Colors
WHITE        = (240, 217, 181)
BLACK        = (181, 136, 99)
HIGHLIGHT    = (246, 246, 105)
HIST_BG      = (30, 30, 30)
HIST_LINE    = (0, 200, 0)
HIST_AXES    = (100, 100, 100)
TEXT_COLOR   = (255, 255, 255)

# Unicode symbols for pieces
def piece_unicode(piece):
    return piece.unicode_symbol() if piece else ""

# Initialize screen
screen = pygame.display.set_mode((BOARD_SIZE, WINDOW_HEIGHT))
pygame.display.set_caption("Play Against Your Agent")
clock = pygame.time.Clock()

# Load font (for chess glyphs and text)
font_path = pygame.font.match_font('dejavusans') or pygame.font.match_font('freesansbold')
if font_path:
    font = pygame.font.Font(font_path, SQ_SIZE - 8)
    small_font = pygame.font.Font(font_path, 18)
else:
    font = pygame.font.SysFont(None, SQ_SIZE - 8)
    small_font = pygame.font.SysFont(None, 18)

# Draw the chess board and pieces
def draw_board(board, selected_sq=None):
    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, 7 - rank)
            color = WHITE if (rank + file) % 2 == 0 else BLACK
            rect = pygame.Rect(file * SQ_SIZE, rank * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, color, rect)
            if selected_sq == sq:
                pygame.draw.rect(screen, HIGHLIGHT, rect, 4)
            piece = board.piece_at(sq)
            if piece:
                symbol = piece_unicode(piece)
                img = font.render(symbol, True, (0, 0, 0))
                screen.blit(img, img.get_rect(center=rect.center))

# Draw the win% history
def draw_history(win_history):
    # Background
    hist_rect = pygame.Rect(0, BOARD_SIZE, BOARD_SIZE, HISTORY_HEIGHT)
    pygame.draw.rect(screen, HIST_BG, hist_rect)
    # Draw axes
    pygame.draw.line(screen, HIST_AXES, (0, BOARD_SIZE + 1), (BOARD_SIZE, BOARD_SIZE + 1))
    pygame.draw.line(screen, HIST_AXES, (0, WINDOW_HEIGHT - 1), (0, BOARD_SIZE), 1)

    if len(win_history) < 2:
        return
    # Scale points
    max_len = BOARD_SIZE  # use full width for history
    data = win_history[-max_len:]
    N = len(data)
    points = []
    for i, w in enumerate(data):
        x = int(i * (BOARD_SIZE / (max_len - 1)))
        # invert y: higher win% = higher on screen
        y = BOARD_SIZE + HISTORY_HEIGHT - int(w * (HISTORY_HEIGHT - 20)) - 10
        points.append((x, y))
    pygame.draw.lines(screen, HIST_LINE, False, points, 2)
    # Draw latest win percent text
    last = data[-1]
    txt = f"Your win%: {last*100:.1f}%"
    txt_img = small_font.render(txt, True, TEXT_COLOR)
    screen.blit(txt_img, (10, BOARD_SIZE + 5))


def main():
    board = chess.Board()
    selected_sq = None
    win_history = []
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if my < BOARD_SIZE:
                    file = mx // SQ_SIZE
                    rank = 7 - (my // SQ_SIZE)
                    clicked_sq = chess.square(file, rank)
                    if selected_sq is None:
                        if board.piece_at(clicked_sq) and board.piece_at(clicked_sq).color == board.turn:
                            selected_sq = clicked_sq
                    else:
                        move = chess.Move(selected_sq, clicked_sq)
                        if move in board.legal_moves:
                            board.push(move)
                            selected_sq = None
                            # Agent's turn
                            if not board.is_game_over():
                                fen = board.fen()
                                best_uci, win_prob = return_next_move(fen)[0]
                                win_history.append(win_prob)
                                board.push(chess.Move.from_uci(best_uci))
                        else:
                            selected_sq = None

        screen.fill((0, 0, 0))
        draw_board(board, selected_sq)
        draw_history(win_history)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()