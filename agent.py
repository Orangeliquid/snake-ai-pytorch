import torch
import random
import numpy as np
import os
from collections import deque
from game import SnakeGameAI_v2, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, get_top_score, update_top_score

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get information about each GPU
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_name(i)
        print(f"GPU {i + 1}: {gpu}")
else:
    print("No GPU available, using CPU.")


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.model.to("cuda")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            if torch.cuda.is_available():
                state_tensor = state_tensor.to("cuda")
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                prediction = self.model(state_tensor)

            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def load_model(self, model_path):
        # Load the model from the specified path
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
            self.model.to("cuda")
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


def train(show_train: bool):
    if show_train:
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
    game = SnakeGameAI_v2(show_game=show_train)
    record = get_top_score()
    agent = Agent()

    if torch.cuda.is_available():
        agent.trainer.model.to("cuda")
        print("Trainer is moved and using GPU")
    else:
        print("Failed to move trainer to GPU...")

    model_path = 'model/model.pth'

    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        agent.load_model(model_path)

        # Check the maximum Q-value for a sample state
        sample_state = np.zeros(11)
        max_q_value = agent.model.get_max_q_value(sample_state)
        print(f"Maximum Q-value for the sample state: {max_q_value}")
    else:
        print("Could not locate model.pth from model_path")

    try:
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    print(f"New score of {score}! New model being saved...")
                    update_top_score(record)
                    agent.trainer.save_model()
                    # Check the maximum Q-value for a sample state
                    sample_state = np.zeros(11)
                    max_q_value = agent.model.get_max_q_value(sample_state)
                    print(f"Maximum Q-value for the sample state is now: {max_q_value}")

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                if show_train:
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / agent.n_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)

    except KeyboardInterrupt:
        print(f"Training interrupted. Saving the model...\nMax Q-Value: {max_q_value}")
        # Call end_save to save the model before exiting
        agent.trainer.end_save(agent)
