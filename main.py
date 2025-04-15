import chess
import chess.svg
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame
# === Global Parameters ===
GAMMA = 0.99
ALPHA = 0.001
EPSILON = 0.9
DECAY = 0.999
MIN_EPSILON = 0.05
EPISODES = 10000
MODEL_FILE = 'dqn_model.pth'
TARGET_UPDATE_FREQ = 100  # Number of episodes between target network updates
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 10000
train_mode = True         # Set to False to load model and play UI
continue_training = False   # Set to True to resume training from saved model
start_episode = 0           # Global variable for tracking the starting episode

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experience Replay Buffer
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# === Neural Network ===
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(773, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Main model and target network for stability
model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=ALPHA)
loss_fn = nn.MSELoss()

# === Tracking Metrics ===
episode_scores = []
episode_moves = []

# === Helper Functions ===
def board_to_tensor(board):
    """
    Encodes a chess board into a tensor of 773 features:
    64 squares x 12 piece types + 5 game state features.
    """
    piece_map = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    plane = np.zeros(773, dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            offset = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
            plane[i * 12 + offset] = piece_map[piece.piece_type]
    plane[-5:] = [
        float(board.turn),
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.can_claim_draw()),
        float(board.is_check())
    ]
    return torch.tensor(plane, dtype=torch.float32, device=device)

def choose_action(board):
    """
    Selects an action using an epsilon-greedy policy.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    if random.random() < EPSILON:
        return random.choice(legal_moves)

    best_score = -float('inf')
    best_move = None
    for move in legal_moves:
        board_copy = board.copy()
        board_copy.push(move)
        state_tensor = board_to_tensor(board_copy).unsqueeze(0)
        q_val = model(state_tensor).item()
        if q_val > best_score:
            best_score = q_val
            best_move = move
    return best_move

def store_transition(state, action, reward, next_state, done):
    """
    Stores a single transition in the replay buffer.
    """
    replay_memory.append((state, action, reward, next_state, done))

def update_model_batch():
    """
    Updates the model using a random mini-batch from the replay memory.
    """
    if len(replay_memory) < BATCH_SIZE:
        return

    batch = random.sample(replay_memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    state_tensors = torch.cat([board_to_tensor(s).unsqueeze(0) for s in states])
    next_tensors = torch.cat([board_to_tensor(s).unsqueeze(0) for s in next_states])
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_vals = model(state_tensors)
    with torch.no_grad():
        next_q_vals = target_model(next_tensors)
    target_q = rewards_tensor + GAMMA * next_q_vals * (1 - dones_tensor)

    loss = loss_fn(q_vals, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def reward_function(board, previous_board=None):
    """
    Returns the reward for the current board.
    For terminal states: returns 1 for win, -1 for loss, or 0 for draws.
    Otherwise, uses a normalized material evaluation.
    If a previous board is provided, uses the delta of material.
    """
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0
    else:
        def evaluate_material(b):
            return sum(
                ((1 if piece.color == chess.WHITE else -1) *
                 {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}[piece.symbol().upper()])
                for piece in b.piece_map().values()
            )
        current_material = evaluate_material(board)
        if previous_board is not None:
            previous_material = evaluate_material(previous_board)
            return (current_material - previous_material) / 39  # normalization factor
        else:
            return current_material / 39

def save_model(episode):
    """
    Saves the current model, optimizer state, and episode number.
    """
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': EPSILON
    }, MODEL_FILE)

def load_model():
    """
    Loads model parameters if a saved model exists.
    Also updates the target network.
    """
    global EPSILON, start_episode
    if os.path.exists(MODEL_FILE):
        checkpoint = torch.load(MODEL_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        EPSILON = checkpoint.get('epsilon', EPSILON)
        start_episode = checkpoint.get('episode', 0)
        model.train()  # Set model to training mode for further training
        target_model.load_state_dict(model.state_dict())
    else:
        start_episode = 0

def save_charts():
    """
    Saves reward and move charts.
    """
    episodes_arr = np.arange(len(episode_scores))
    plt.figure()
    plt.plot(episodes_arr, episode_scores, label='Game Outcome')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Game Outcome per Episode')
    plt.grid(True)
    plt.savefig('score_chart.png')
    plt.close()

    plt.figure()
    plt.plot(episodes_arr, episode_moves, label='Moves per Game', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Number of Moves')
    plt.title('Moves per Episode')
    plt.grid(True)
    plt.savefig('moves_chart.png')
    plt.close()

# === Training ===
def train_agents():
    global EPSILON
    for episode in range(start_episode, EPISODES + start_episode):
        print(f"\rTraining Episode: {episode+1}/{EPISODES+start_episode}", end="")
        board = chess.Board()
        moves = 0
        previous_board = board.copy()  # For delta reward calculation
        while not board.is_game_over():
            state = board.copy()
            action = choose_action(board)
            if action is None:
                break
            board.push(action)
            reward = reward_function(board, previous_board)
            next_state = board.copy()
            done = board.is_game_over()
            store_transition(state, action, reward, next_state, done)
            update_model_batch()
            moves += 1
            previous_board = state  # update previous board
            if done:
                break
        EPSILON = max(MIN_EPSILON, EPSILON * DECAY)
        final_reward = reward_function(board)
        episode_scores.append(final_reward)
        episode_moves.append(moves)

        if (episode - start_episode) % 100 == 0 and episode > start_episode:
            save_charts()
            save_model(episode)
            target_model.load_state_dict(model.state_dict())
    save_model(episode)
    print("\nTraining complete and model saved.")

# === Pygame-based UI for Human vs Agent ===
def play_human():
    """
    Provides a Pygame UI to play against the trained agent.
    Use mouse clicks to move pieces:
      - Click on a white piece to select it.
      - Click on a destination square to move.
    After your move, the agent (playing Black) moves.
    """
    import pygame
    pygame.init()
    board = chess.Board()
    screen_size = 120
    square_size = screen_size // 8
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Chess: Human vs Agent")
    font = pygame.font.SysFont("Arial", square_size // 2)
    selected_square = None
    running = True

    while running and not board.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                file = x // square_size
                rank = 7 - (y // square_size)  # Adjust rank: bottom of board is rank 0
                square = chess.square(file, rank)
                if selected_square is None:
                    piece = board.piece_at(square)
                    # Allow selection only if there's a piece and it's white (human)
                    if piece and piece.color == chess.WHITE:
                        selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                        # After human move, let the agent (Black) move if game not over
                        if not board.is_game_over():
                            agent_move = choose_action(board)
                            if agent_move is not None:
                                board.push(agent_move)
                    else:
                        selected_square = None

        draw_board(board, screen, square_size, font, selected_square)
        pygame.display.flip()

    print("Game over!", board.result())
    pygame.quit()

def draw_board(board, screen, square_size, font, selected_square):
    """
    Draws the chess board and pieces on the Pygame screen using plain text characters.
    """
    white_color = (240, 217, 181)
    black_color = (181, 136, 99)
    highlight_color = (106, 246, 76)
    
    for rank in range(8):
        for file in range(8):
            square_color = white_color if (file + rank) % 2 == 0 else black_color
            rect = pygame.Rect(file * square_size, (7 - rank) * square_size, square_size, square_size)
            pygame.draw.rect(screen, square_color, rect)
            
            current_square = chess.square(file, rank)
            if selected_square is not None and selected_square == current_square:
                pygame.draw.rect(screen, highlight_color, rect, 3)
                
            piece = board.piece_at(current_square)
            if piece is not None:
                # Use plain characters: uppercase for white, lowercase for black.
                piece_text = piece.symbol()  # e.g., 'K' for white king, 'k' for black king
                # Choose text color for visibility (e.g., white pieces: black text; black pieces: white text)
                text_color = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                text_surface = font.render(piece_text, True, text_color)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)

if __name__ == '__main__':
    if train_mode:
        if continue_training:
            load_model()  # Loads state and sets start_episode
        train_agents()
    else:
        load_model()
        play_human()
