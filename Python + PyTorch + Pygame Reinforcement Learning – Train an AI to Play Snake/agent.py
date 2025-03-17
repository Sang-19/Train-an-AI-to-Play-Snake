import torch  # PyTorch for deep learning
import random  # Random number generation
import numpy as npy  # Numerical computations
from collections import deque  # Double-ended queue for replay memory
from game import SnakeGameAI, Direction, Point  # Import game logic
from model import Linear_QNet, QTrainer  # Import neural network model and trainer
from helper import plot  # Import helper function for plotting

MAX_MEMORY = 100_000  # Maximum memory size for replay buffer
BATCH_SIZE = 1000  # Training batch size
LR = 0.001  # Learning rate for training

class Agent:  # AI Agent class
    def __init__(self):
        self.n_game = 0  # Number of games played
        self.epsilon = 0  # Exploration-exploitation tradeoff
        self.gamma = 0.9  # Discount rate for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay memory
        self.model = Linear_QNet(11, 256, 3)  # Neural network (11 inputs, 256 hidden, 3 outputs)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Initialize trainer

    def get_state(self, game_state):
        head = game_state.snake[0]  # Get snake's head position
        point_l = Point(head.x - 20, head.y)  # Left of head
        point_r = Point(head.x + 20, head.y)  # Right of head
        point_u = Point(head.x, head.y - 20)  # Above head
        point_d = Point(head.x, head.y + 20)  # Below head
        
        dir_l = game_state.direction == Direction.LEFT  # Is moving left?
        dir_r = game_state.direction == Direction.RIGHT  # Is moving right?
        dir_u = game_state.direction == Direction.UP  # Is moving up?
        dir_d = game_state.direction == Direction.DOWN  # Is moving down?

        state = [
            # Danger straight
            (dir_r and game_state.is_collision(point_r)) or
            (dir_l and game_state.is_collision(point_l)) or
            (dir_u and game_state.is_collision(point_u)) or
            (dir_d and game_state.is_collision(point_d)),

            # Danger right
            (dir_u and game_state.is_collision(point_r)) or
            (dir_d and game_state.is_collision(point_l)) or
            (dir_l and game_state.is_collision(point_u)) or
            (dir_r and game_state.is_collision(point_d)),

            # Danger left
            (dir_d and game_state.is_collision(point_r)) or
            (dir_u and game_state.is_collision(point_l)) or
            (dir_r and game_state.is_collision(point_u)) or
            (dir_l and game_state.is_collision(point_d)),

            # Movement direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            game_state.food.x < game_state.head.x,  # Food is left
            game_state.food.x > game_state.head.x,  # Food is right
            game_state.food.y < game_state.head.y,  # Food is up
            game_state.food.y > game_state.head.y  # Food is down
        ]
        return npy.array(state, dtype=int)  # Convert to numpy array (int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Save experience to memory

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Get random batch from memory
        else:
            mini_sample = self.memory  # Use all stored experiences

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # Unzip batch
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Train using batch

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)  # Train on single move

    def get_action(self, state):
        self.epsilon = 80 - self.n_game  # Reduce randomness over time
        final_move = [0, 0, 0]  # Initialize move array

        if random.randint(0, 200) < self.epsilon:  # Exploration: Choose random move
            move = random.randint(0, 2)  
            final_move[move] = 1  
        else:  # Exploitation: Choose best move from model
            state0 = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
            prediction = self.model(state0)  # Get model's prediction
            move = torch.argmax(prediction).item()  # Get action with highest Q-value
            final_move[move] = 1  

        return final_move  # Return action

def train():  # Main training loop
    plot_scores = []  # List to store scores
    plot_mean_scores = []  # List to store mean scores
    total_score = 0  # Initialize total score
    record = 0  # Track highest score
    agent = Agent()  # Create AI agent
    game = SnakeGameAI()  # Start game instance

    while True:  # Infinite loop to train AI
        state_old = agent.get_state(game)  # Get current state
        final_move = agent.get_action(state_old)  # AI decides action
        reward, done, score = game.play_step(final_move)  # Perform action
        state_new = agent.get_state(game)  # Get new state

        agent.train_short_memory(state_old, final_move, reward, state_new, done)  # Train with short-term memory
        agent.remember(state_old, final_move, reward, state_new, done)  # Save experience

        if done:  # If game over
            game.reset()  # Reset game
            agent.n_game += 1  # Increase game count
            agent.train_long_memory()  # Train with long-term memory

            if score > record:  # If new high score
                record = score  
                agent.model.save()  # Save model

            print('Game', agent.n_game, 'Score', score, 'Record:', record)  # Print game stats

            plot_scores.append(score)  # Store score
            total_score += score  
            mean_score = total_score / agent.n_game  # Calculate mean score
            plot_mean_scores.append(mean_score)  
            plot(plot_scores, plot_mean_scores)  # Plot performance graph

if __name__ == '__main__':
    train()  # Start training when script runs
