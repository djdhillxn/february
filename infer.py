# infer.py
import torch
from hangman.model import HangmanLSTMNet
from hangman.game import HangmanPlayer

def load_model(model_path, hidden_dim):
    model = HangmanLSTMNet(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    model = load_model('hangman_model.pth', hidden_dim=128)
    
    while True:
        word = input("Enter a word for testing: ")
        player = HangmanPlayer(word, model)
        player.run()
        player.play_by_play()

        if input("Try another word? (y/n): ") != 'y':
            break

if __name__ == '__main__':
    main()

