import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network module
import torch.nn.functional as F  # Activation functions
import torch.optim as optim  # Optimization algorithms
import os  # For file operations
import numpy as npy  # NumPy for numerical operations

# Define a neural network for Q-learning
class Linear_QNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)  # First hidden layer
    self.linear2 = nn.Linear(hidden_size, output_size)  # Output layer

  def forward(self, x):
    x = F.relu(self.linear1(x))  # Apply ReLU activation to hidden layer
    x = self.linear2(x)  # Output layer (no activation, raw scores)
    return x

  def save(self, file_name='model.pth'):
    model_folder = './model'  # Folder to save the model
    if not os.path.exists(model_folder):  # Create folder if it doesn't exist
      os.makedirs(model_folder)
    file_name = os.path.join(model_folder, file_name)  # Full file path
    torch.save(self.state_dict(), file_name)  # Save model parameters

# Q-learning trainer for training the neural network
class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr  # Learning rate
    self.gamma = gamma  # Discount factor for future rewards
    self.model = model  # Q-network model
    self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Adam optimizer
    self.criterion = nn.MSELoss()  # Mean Squared Error loss function

  def train_step(self, state, action, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
    next_state = torch.tensor(next_state, dtype=torch.float)  # Convert next state
    action = torch.tensor(action, dtype=torch.long)  # Convert action to tensor
    reward = torch.tensor(reward, dtype=torch.float)  # Convert reward to tensor

    # Check if it's a single data point, and convert to batch format if needed
    if len(state.shape) == 1:
      state = torch.unsqueeze(state, 0)  # Add batch dimension
      next_state = torch.unsqueeze(next_state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      done = (done, )  # Convert done flag into tuple

    # Get predicted Q values for current state
    pred = self.model(state)

    # Clone the predictions to modify targets
    target = pred.clone()
    for idx in range(len(done)):
      Q_new = reward[idx]  # Start with immediate reward
      if not done[idx]:  # If game is not over, update Q value
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

      # Update the correct action's Q-value in target
      target[idx][torch.argmax(action[idx]).item()] = Q_new

    self.optimizer.zero_grad()  # Clear gradients
    loss = self.criterion(target, pred)  # Compute loss
    loss.backward()  # Backpropagation
    self.optimizer.step()  # Update model weights
