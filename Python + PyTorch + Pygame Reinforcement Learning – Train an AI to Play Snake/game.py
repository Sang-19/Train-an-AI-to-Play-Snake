# Required library imports
import pygame                  # Game development library
import random                 # For random food placement
from enum import Enum         # For creating direction constants
from collections import namedtuple  # For creating Point structure
import numpy as npy          # For AI action handling

# Initialize pygame and set up font
pygame.init()
font = pygame.font.Font('arial.ttf', 20)  # Load font for score display

# Define possible snake movement directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
# Create Point structure for x,y coordinates
Point = namedtuple('Point', 'x, y')

# Define colors for game elements
WHITE = (255, 255, 255)  # Score text color
RED = (200,0,0)         # Food color
BLUE1 = (0, 0, 255)     # Snake outer color
GREEN = (0, 255, 0)   # Snake inner color
BLACK = (0,0,0)         # Background color

# Game constants
BLOCK_SIZE = 20         # Size of snake segments and food
SPEED = 20             # Game speed (frames per second)

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w                  # Game window width
        self.h = h                  # Game window height
        self.display = pygame.display.set_mode((self.w, self.h))  # Create game window
        pygame.display.set_caption('Snake')  # Set window title
        self.clock = pygame.time.Clock()    # For controlling game speed
        self.reset()                        # Initialize game state
        
    def reset(self):
        self.direction = Direction.RIGHT    # Initial snake direction
        self.head = Point(self.w/2, self.h/2)  # Start snake at center
        # Create initial snake body (3 segments)
        self.snake = [self.head, 
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0              # Reset score
        self.food = None           # Food position
        self._place_food()         # Place initial food
        self.frame_iteration = 0   # Frame counter for timeout
        
    def _place_food(self):
        # Calculate random position for food
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # Recursively try again if food spawns on snake
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1   # Increment frame counter
        # Handle quit event
        for event in pygame.event.get():    #Retrieves a list of events (e.g., key presses, mouse clicks, window close, etc.).
            if event.type == pygame.QUIT:
                pygame.quit()   #is triggered when the user clicks the "X" button on the game window.
                quit()
        
        self._move(action)          # Move snake based on AI action
        self.snake.insert(0, self.head)  # Update snake body
        
        # Check game over conditions
        reward = 0   #The reward variable starts at 0 (neutral reward)
        game_over = False  #game_over is initially False, meaning the game is still running.
        
        
        
        
        
        #revents the AI from stalling (taking too long to make a move).
        #snake doesn’t eat food for too long, it automatically loses 
        #The timeout increases as the snake grows (100 frames per body segment)        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10   #Gives a negative reward (-10) to discourage the AI from losing.
            return reward, game_over, self.score
            
        # Handle food collection
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()        # Remove tail if no food eaten
        #If the snake did not eat food, remove the last segment of the snake.
        #This maintains the snake's length unless it eats food.





        self._update_ui()           # Update game display
        self.clock.tick(SPEED)      # Control game speed
        return reward, game_over, self.score
    
    def is_collision(self, pt = None):
        if pt is None:
            pt = self.head
        # Check boundary collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check self-collision
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)    # Clear screen
        
        # Draw snake body
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()       # Update display
        
    def _move(self, action):
        # Define possible directions in clockwise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        # Interpret AI action
        if npy.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]     # Continue straight
        elif npy.array_equal(action, [0, 1, 0]):
            next_dir = (idx + 1) % 4      # Turn right
            new_dir = clock_wise[next_dir]
        else:  # [0, 0, 1]
            next_dir = (idx - 1) % 4      # Turn left
            new_dir = clock_wise[next_dir]
            
        self.direction = new_dir
        
        # Update head position based on direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)    # Update head position
            

