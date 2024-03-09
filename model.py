import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import shutil


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure that the input tensor is on the same device as the model's parameters
        if not self.linear1.weight.device == x.device:
            x = x.to(self.linear1.weight.device)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def get_max_q_value(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        if next(self.parameters()).is_cuda:
            state_tensor = state_tensor.cuda()

        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            prediction = self(state_tensor)

        max_q_value, _ = torch.max(prediction, dim=1)
        return max_q_value.item()


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.games_played = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name)

    # def save(self, file_name='model.pth'):
    #     model_folder_path = './model'
    #     if not os.path.exists(model_folder_path):
    #         os.makedirs(model_folder_path)
    #
    #     file_name = os.path.join(model_folder_path, file_name)
    #     torch.save(self.state_dict(), file_name)

    def load_model(self, file_name='model.pth'):
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}")

    def end_save(self, agent, file_name='model.pth'):
        # Update the games_played attribute in the QTrainer
        self.games_played = agent.n_games

        # Save the current model
        self.save_model(file_name)

        # Optionally, create a duplicate if the agent played over 200 games
        if self.games_played > 100:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            duplicate_file_name = f"{file_name.split('.pth')[0]}_duplicate_{timestamp}.pth"
            duplicate_file_path = os.path.join('./model', duplicate_file_name)

            # Copy the current model.pth to create a duplicate
            shutil.copyfile(os.path.join('./model', file_name), duplicate_file_path)
