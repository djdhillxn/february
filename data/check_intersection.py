import sys
import tqdm

def read_words_from_file(filename):
    # Read all the words from a given file
    with open(filename, 'r') as file:
        words = file.read().splitlines()
    return words

def find_common_words(file1, file2):
    # Read words from both files
    words_file1 = read_words_from_file(file1)
    words_file2 = read_words_from_file(file2)

    # Convert lists to sets for efficient intersection
    set_file1 = set(words_file1)
    set_file2 = set(words_file2)

    # Find intersection of both sets
    common_words = set_file1.intersection(set_file2)

    return common_words

# Using command line arguments for file paths
if len(sys.argv) != 3:
    print("Usage: python script.py <file1_path> <file2_path>")
    sys.exit(1)

file1_path = sys.argv[1]
file2_path = sys.argv[2]

# Read and count words from both files
words_file1 = read_words_from_file(file1_path)
words_file2 = read_words_from_file(file2_path)

# Find common words
common_words = find_common_words(file1_path, file2_path)

# Print the result
print(f"Number of words in file1: {len(words_file1)}")
print(f"Number of words in file2: {len(words_file2)}")
print(f"Number of common words: {len(common_words)}")

