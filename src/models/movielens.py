import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class RecommendationSystemModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.2,   
    ):
        super(RecommendationSystemModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_size)
        self.movie_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_size)

        # Hidden layers
        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, users, movies):
        # Embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        # Concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_embedded], dim=1)

        # Pass through hidden layers with ReLU activation and dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output

    def predict(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[np.ndarray, np.ndarray]:
        self.eval()

        predictions = []
        targets = []
        with torch.no_grad():
            for _, test_data in tqdm(enumerate(loader)):
                output = self.forward(test_data["users"].to(device), test_data["items"].to(device))
                output = output.squeeze()
                predictions.extend(output.cpu().numpy())
                targets.extend(test_data["ratings"].numpy())

        return np.array(predictions), np.array(targets)
