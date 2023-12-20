# model.py
import torch
import torch.nn as nn

class HangmanLSTMNet(nn.Module):
    def __init__(self, hidden_dim, lstm_layers=1):
        super(HangmanLSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(27, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 26, 26)

    def forward(self, obscured_word, previous_guesses):
        lstm_out, _ = self.lstm(obscured_word)
        final_lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((final_lstm_out, previous_guesses), dim=1)
        out = self.fc(combined)
        return out

    def guess(self, current_word, previous_guesses):
        # Process current_word and previous_guesses to create tensors
        obscured_word_tensor = self._encode_obscured_word(current_word)
        prev_guesses_tensor = self._encode_previous_guesses(previous_guesses)

        with torch.no_grad():
            output = self.forward(obscured_word_tensor.unsqueeze(0), prev_guesses_tensor.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)
            probabilities[0, previous_guesses] = 0  # Zero out already guessed letters
            return torch.argmax(probabilities).item()

    def _encode_obscured_word(self, current_word):
        # Convert the obscured word into a tensor
        return torch.tensor([[ord(c) - 97 if c != '_' else 26 for c in current_word]], dtype=torch.float32)

    def _encode_previous_guesses(self, previous_guesses):
        # Convert the list of guessed letters into a tensor
        guesses_tensor = torch.zeros(26, dtype=torch.float32)
        for guess in previous_guesses:
            guesses_tensor[ord(guess) - 97] = 1
        return guesses_tensor

