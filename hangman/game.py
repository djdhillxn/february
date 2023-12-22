# game.py
import torch
import numpy as np

class HangmanPlayer:
    def __init__(self, word, model, lives=6, device='cpu'):
        self.device = device
        self.original_word = word.lower()
        self.full_word = [ord(i) - 97 for i in word]
        self.letters_guessed = set()
        self.letters_remaining = set(self.full_word)
        self.lives_left = lives
        self.obscured_words_seen = []
        self.letters_previously_guessed = []
        self.guesses = []
        self.correct_responses = []
        self.model = model

    def encode_obscured_word(self):
        word = [i if i in self.letters_guessed else 26 for i in self.full_word]
        obscured_word = torch.zeros((len(word), 27), dtype=torch.float32, device=self.device)
        for i, j in enumerate(word):
            obscured_word[i, j] = 1
        return obscured_word

    def encode_previous_guesses(self):
        guess = torch.zeros(26, dtype=torch.float32, device=self.device)
        for i in self.letters_guessed:
            guess[i] = 1
        return guess
    
    def encode_correct_responses(self):
        response = torch.zeros(26, dtype=torch.float32, device=self.device)
        for i in self.letters_remaining:
            response[i] = 1.0
        if self.letters_remaining:
            response /= response.sum()  # Normalize the response vector
        return response

    def store_guess_and_result(self, guess):
        self.obscured_words_seen.append(self.encode_obscured_word())
        self.letters_previously_guessed.append(self.encode_previous_guesses())
        # Add the guessed letter to the set of guessed letters if not already guessed
        if guess not in self.letters_guessed:
            self.letters_guessed.add(guess)
            # Check if the guess is correct
            if guess in self.letters_remaining:
                self.letters_remaining.remove(guess)
            else:
                self.lives_left -= 1
        # Record the updated correct responses
        correct_responses = self.encode_correct_responses()
        self.correct_responses.append(correct_responses)
        return

    def run(self):
        while self.lives_left > 0 and len(self.letters_remaining) > 0:
            obscured_word_tensor = self.encode_obscured_word().unsqueeze(0)
            prev_guesses_tensor = self.encode_previous_guesses().unsqueeze(0)

            valid_guesses_mask = torch.ones(26, dtype=torch.float32, device=self.device)
            for g in self.letters_guessed:
                valid_guesses_mask[g] = 0

            model_output = self.model(obscured_word_tensor, prev_guesses_tensor).squeeze(0)
            model_output[valid_guesses_mask == 0] = -float('inf')  # Invalidate repeated guesses
            guess = torch.argmax(model_output).item()

            self.store_guess_and_result(guess)

        return (torch.stack(self.obscured_words_seen),
                torch.stack(self.letters_previously_guessed),
                torch.stack(self.correct_responses))
    """
    def run(self):
        iteration = 0
        while self.lives_left > 0 and len(self.letters_remaining) > 0:
            iteration += 1
            #print(f"\n--- Iteration {iteration} ---")
            #print(f"Current word: '{self.original_word}'")
            obscured_word_repr = ''.join([chr(i + 97) if i != 26 else '_' for i in self.encode_obscured_word().argmax(axis=1)])
            #print(f"Obscured word: {obscured_word_repr}")
            #print(f"Lives left: {self.lives_left}")
            #print(f"Letters guessed: {''.join(sorted([chr(i + 97) for i in self.letters_guessed]))}")
            # Generate guess from model
            obscured_word_tensor = self.encode_obscured_word().unsqueeze(0)
            prev_guesses_tensor = self.encode_previous_guesses().unsqueeze(0)
            # Prevent model from repeating guesses
            valid_guesses_mask = torch.ones(26, dtype=torch.float32, device=self.device)
            for g in self.letters_guessed:
                valid_guesses_mask[g] = 0
            model_output = self.model(obscured_word_tensor, prev_guesses_tensor).squeeze(0)
            model_output[valid_guesses_mask == 0] = -float('inf')  # Invalidate repeated guesses
            guess = torch.argmax(model_output).item()
            #print(f"Model's guess: {chr(guess + 97)}")
            # Store the result and update the game state
            self.store_guess_and_result(guess)
        return (torch.stack(self.obscured_words_seen),
                torch.stack(self.letters_previously_guessed),
                torch.stack(self.correct_responses))
    """

    def show_words_seen(self):
        for word in self.obscured_words_seen:
            print(''.join([chr(i + 97) if i != 26 else '_' for i in word.argmax(axis=1)]))

    def show_guesses(self):
        for guess in self.guesses:
            print(chr(guess + 97))

    def play_by_play(self):
        print('Hidden word was "{}"'.format(self.original_word))
        for i in range(len(self.guesses)):
            word_seen = ''.join([chr(i + 97) if i != 26 else '_' for i in np.array(self.obscured_words_seen[i]).argmax(axis=1)])
            print('Guessed {} after seeing "{}"'.format(chr(self.guesses[i] + 97),
                                                        word_seen))
    
    def evaluate_performance(self):
        ended_in_success = self.lives_left > 0
        letters_in_word = set([i for i in self.original_word])
        correct_guesses = len(letters_in_word) - len(self.letters_remaining)
        incorrect_guesses = len(self.guesses) - correct_guesses
        return (ended_in_success, correct_guesses, incorrect_guesses, letters_in_word)


