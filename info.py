from hangman.model import HangmanGRUNet
model = HangmanGRUNet(hidden_dim=128, gru_layers=3, device='cpu') # replace with your actual model and parameters
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_parameters}")

