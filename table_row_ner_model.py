import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SimpleNERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

# Example usage:
vocab_size = 10000  # Replace with the actual size of your vocabulary
embedding_dim = 50
hidden_dim = 64
num_classes = 10  # Replace with the actual number of NER classes

model = SimpleNERModel(vocab_size, embedding_dim, hidden_dim, num_classes)

# Sample input data (sequence of token indices)
input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# Forward pass
logits = model(input_ids)
print("Model Output Shape:", logits.shape)
