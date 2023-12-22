# model.py
import torch
import torch.nn as nn

class HangmanGRUNet(nn.Module):
    def __init__(self, hidden_dim, gru_layers=1, device='cpu'):
        super(HangmanGRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(27, hidden_dim, num_layers=gru_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_dim + 26, 26).to(device)
        self.device = device

    def forward(self, obscured_word, previous_guesses):
        gru_out, _ = self.gru(obscured_word)
        final_gru_out = gru_out[:, -1, :]
        combined = torch.cat((final_gru_out, previous_guesses), dim=1)
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
        return torch.tensor([[ord(c) - 97 if c != '_' else 26 for c in current_word]], dtype=torch.float32, device=self.device)

    def _encode_previous_guesses(self, previous_guesses):
        # Convert the list of guessed letters into a tensor
        guesses_tensor = torch.zeros(26, dtype=torch.float32, device=self.device)
        for guess in previous_guesses:
            guesses_tensor[ord(guess) - 97] = 1
        return guesses_tensor

