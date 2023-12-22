# train.py
import torch
from hangman.model import HangmanGRUNet
from hangman.game import HangmanPlayer
from tqdm import tqdm
import random
import pandas as pd 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def validate_model(model, data):
    results = []
    progress_bar = tqdm(data, desc='Validation', unit='word')
    for word in progress_bar:
        player = HangmanPlayer(word, model, device)
        _ = player.run()
        results.append(player.evaluate_performance())
    df = pd.DataFrame(results, columns=['won', 'num_correct', 'num_incorrect', 'letters'])
    return df


def train_model(model, train_data, val_data, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # Negative Log-Likelihood Loss
    for epoch in range(epochs):
        random.shuffle(train_data)
        epoch_loss = 0
        progress_bar = tqdm(train_data, desc=f'Epoch {epoch + 1}/{epochs}', unit='word')
        # Training
        for word in progress_bar:
            player = HangmanPlayer(word, model, device=device)
            words_seen, previous_letters, correct_responses = player.run()
            optimizer.zero_grad()
            for i in range(len(words_seen)):
                output = model(words_seen[i].unsqueeze(0), previous_letters[i].unsqueeze(0))
                loss = criterion(output, correct_responses[i].unsqueeze(0).to(device))
                loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            progress_bar.set_postfix(loss=epoch_loss / (len(train_data) * (i + 1)))
            # Verbose Output
            #print(f"Processed word: '{word}', Loss: {loss.item():.4f}")
        
        # Print training results
        print(f'Epoch {epoch + 1} completed, Average Loss: {epoch_loss / len(train_data):.4f}')
        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_result_df = validate_model(model, val_data)

        # Print validation results
        print(f'Validation Results - Epoch {epoch + 1}:')
        print(f'- Averaged {val_result_df["num_correct"].mean():.1f} correct and {val_result_df["num_incorrect"].mean():.1f} incorrect guesses per game')
        print(f'- Won {100 * val_result_df["won"].sum() / len(val_result_df):.1f}% of games played')
        model.train()  # Set the model back to training mode


    torch.save(model.state_dict(), 'hangman_model.pth')


def load_data(filepath):
    with open(filepath, 'r') as file:
        words = [line.strip() for line in file]
    return words

def main():
    print("Loading training data...")
    train_words = load_data('./data/words_250000_train.txt')
    print(f"Loaded {len(train_words)} words for training.") 

    print("Loading testing data...")
    test_words = load_data('./data/words_test.txt')
    print(f"Loaded {len(test_words)} words for testing.")

    print("Initializing model...")
    model = HangmanGRUNet(hidden_dim=128, gru_layers=1, device=device)
    print("Model initialized.")

    print("Starting training...")
    train_model(model, train_words[:100000], test_words[:100000], epochs=3, learning_rate=0.001)
    print("Training completed.")

    print("Saving model...")
    torch.save(model.state_dict(), 'hangman_model.pth')
    print("Model saved as 'hangman_model.pth'.")

if __name__ == '__main__':
    main()

