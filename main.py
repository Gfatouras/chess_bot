import chess
import chess.svg
import random
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# === Global Parameters ===
GAMMA = 0.99
ALPHA = 0.001
EPSILON = 0.9
DECAY = 0.999
MIN_EPSILON = 0.05
EPISODES = 10000
MODEL_FILE = 'dqn_model.pth'
train = True  # Set to False to load model and play UI
continue_training = False  # Set to True to resume training from saved model

# === Neural Network ===
class DQN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(773, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=ALPHA)
loss_fn = nn.MSELoss()

# === Tracking Metrics ===
episode_scores = []
episode_moves = []

# === Helper Functions ===
def board_to_tensor(board):
	# Encode board into a tensor representation (773 features)
	# 64 squares x 12 pieces + 5 game state bits
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
	plane[-5:] = [board.turn, board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE), board.can_claim_draw(), board.is_check()]
	return torch.tensor(plane, dtype=torch.float32)

def choose_action(board):
	legal_moves = list(board.legal_moves)
	if not legal_moves:
		return None
	if random.random() < EPSILON:
		return random.choice(legal_moves)
	best_score = -float('inf')
	best_move = None
	state_tensor = board_to_tensor(board)
	for move in legal_moves:
		board_copy = board.copy()
		board_copy.push(move)
		q_val = model(board_to_tensor(board_copy).unsqueeze(0)).item()
		if q_val > best_score:
			best_score = q_val
			best_move = move
	return best_move

def update_model(state, action, reward, next_state, done):
	state_tensor = board_to_tensor(state).unsqueeze(0)
	next_tensor = board_to_tensor(next_state).unsqueeze(0)
	q_val = model(state_tensor)
	with torch.no_grad():
		target_q = reward if done else reward + GAMMA * model(next_tensor).item()
	loss = loss_fn(q_val, torch.tensor([[target_q]], dtype=torch.float32))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

def reward_function(board):
	if board.is_checkmate():
		return 1 if board.turn == chess.BLACK else -1
	elif board.is_stalemate() or board.is_insufficient_material():
		return 0
	else:
		material = sum(
			(piece.symbol().isupper() * 2 - 1) * {
				'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0
			}[piece.symbol().upper()] for piece in board.piece_map().values()
		)
		return material / 39  # Normalize

def save_model():
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epsilon': EPSILON
	}, MODEL_FILE)


def load_model():
	global EPSILON
	if os.path.exists(MODEL_FILE):
		checkpoint = torch.load(MODEL_FILE)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		EPSILON = checkpoint.get('epsilon', EPSILON)
		model.eval()


def save_charts():
	episodes = np.arange(len(episode_scores))
	plt.figure()
	plt.plot(episodes, episode_scores, label='Game Result')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Game Outcome per Episode')
	plt.grid(True)
	plt.savefig('score_chart.png')
	plt.close()

	plt.figure()
	plt.plot(episodes, episode_moves, label='Moves per Game', color='orange')
	plt.xlabel('Episode')
	plt.ylabel('Number of Moves')
	plt.title('Moves per Episode')
	plt.grid(True)
	plt.savefig('moves_chart.png')
	plt.close()

# === Training ===
def train_agents():
	global EPSILON
	for episode in range(EPISODES):
		print(f"\rTraining Episode: {episode+1}/{EPISODES}", end="")
		board = chess.Board()
		moves = 0
		while not board.is_game_over():
			state = board.copy()
			action = choose_action(board)
			if action is None:
				break
			board.push(action)
			reward = reward_function(board)
			next_state = board.copy()
			done = board.is_game_over()
			update_model(state, action, reward, next_state, done)
			moves += 1
			if done:
				break
		EPSILON = max(MIN_EPSILON, EPSILON * DECAY)
		final_reward = reward_function(board)
		episode_scores.append(final_reward)
		episode_moves.append(moves)
		if episode % 100 == 0 and episode > 0:
			save_charts()
			save_model()
	save_model()
	print("\nTraining complete and model saved.")

# === UI for Human vs Agent ===
def play_human():
	import chess.svg
	import webbrowser
	import tempfile
	load_model()
	board = chess.Board()
	while not board.is_game_over():
		print(board)
		if board.turn == chess.WHITE:
			move_uci = input("Your move (in UCI format, e.g., e2e4): ")
			try:
				move = chess.Move.from_uci(move_uci)
				if move in board.legal_moves:
					board.push(move)
				else:
					print("Illegal move.")
			except Exception:
				print("Invalid input.")
		else:
			move = choose_action(board)
			print(f"Agent move: {move}")
			board.push(move)
		with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
			f.write(chess.svg.board(board, size=400))
			webbrowser.open(f.name)
	print("Game over!", board.result())

if __name__ == '__main__':
	if train:
		if continue_training:
			load_model()
		train_agents()
	else:
		load_model()
		play_human()
