# 🐍 AI Snake Game - Reinforcement Learning

This project is an **AI-powered Snake Game** built using **Python, PyTorch, and Reinforcement Learning (Deep Q-Learning)**. The AI learns to play the game efficiently over time by maximizing rewards and avoiding collisions.

## 🚀 Features
- 🧠 **Deep Q-Learning Agent** for smart decision-making.
- 🎮 **PyGame-based UI** for real-time game visualization.
- 📈 **Performance tracking** with graphical results.
- 🔄 **Model saving/loading** for continuous training.

## 📂 Project Structure
```
├── model.py        # Defines the Deep Q-Network (DQN) model
├── game.py         # Implements the Snake game mechanics using PyGame
├── agent.py        # AI Agent that trains using Deep Q-Learning
├── helper.py       # Utility functions (e.g., plotting results)
├── train.py        # Main script to train the AI
├── model/          # Folder where trained models are saved
├── README.md       # Project documentation
```

## 🛠️ Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
(If `requirements.txt` is missing, install manually:)
```bash
pip install pygame torch numpy matplotlib
```

### 2️⃣ Run the Training Script
```bash
python train.py
```

### 3️⃣ Watch the AI Play
Once trained, the AI will improve over time and learn to survive longer.

## 📊 How the AI Works
The AI uses a **Deep Q-Network (DQN)** with the following steps:
1. **State Representation**: The game environment is converted into a feature set.
2. **Action Selection**: The AI chooses an action (move left, right, or straight) based on past experiences.
3. **Reward System**: The AI gets rewarded for eating food and penalized for dying.
4. **Experience Replay**: Uses past experiences to improve decision-making.

## 🎯 Future Improvements
✅ Train with **Convolutional Neural Networks (CNNs)** for image-based learning.
✅ Implement **Evolutionary Algorithms** for better optimization.
✅ Add **Multiplayer Mode** or **Advanced Graphics**.

## 📜 License
This project is open-source under the **MIT License**.

---
💡 **Made with Python & AI for fun and learning!** 🚀

