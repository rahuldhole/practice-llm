import random
import time

class BatchLoader:
    """
    A custom iterable class that implements the Iterator Protocol.
    Takes a dataset and yields mini-batches of a specified size.
    Similar to a basic PyTorch DataLoader.
    """
    def __init__(self, data, batch_size, shuffle=False):
        self.data = list(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0

    def __iter__(self):
        """
        Returns the iterator object itself.
        By setting index to 0 (and shuffling if requested), we reset state
        every time a new loop is started.
        """
        self.index = 0
        if self.shuffle:
            random.shuffle(self.data)
        return self

    def __next__(self):
        """
        Returns the next batch. Raises StopIteration when done.
        """
        if self.index >= len(self.data):
            raise StopIteration
        
        end_idx = min(self.index + self.batch_size, len(self.data))
        batch = self.data[self.index:end_idx]
        self.index = end_idx
        return batch


# Generator Functions (using yield)

def stream_tokens(text):
    """
    A simple generator that yields words (tokens) from text one by one.
    Demonstrates lazy evaluation: the entire list of words is never loaded into
    memory at once.
    """
    word = []
    for char in text:
        if char.isspace():
            if word:
                time.sleep(0.1)  # Simulate 100ms network/generation delay
                yield "".join(word)
                word = []
        else:
            word.append(char)
    if word:
        time.sleep(0.1)  # Simulate 100ms network/generation delay
        yield "".join(word)


# Generator Pipelines

def clean_filter(tokens):
    """
    Generator pipeline stage 1: Cleans punctuation and lowercase tokens.
    """
    for token in tokens:
        cleaned = "".join(c for c in token if c.isalnum()).lower()
        if cleaned:
            yield cleaned


def bigrams(tokens):
    """
    Generator pipeline stage 2: Takes a stream of tokens and yields sliding window bigrams.
    """
    prev = None
    for token in tokens:
        if prev is not None:
            yield (prev, token)
        prev = token


if __name__ == "__main__":
    print("--- Generators & Iterators Demo ---")
    
    # 1. Custom Iterator (BatchLoader)
    dataset = list(range(1, 11))  # 1 to 10
    loader = BatchLoader(dataset, batch_size=3, shuffle=True)
    
    print("Epoch 1 (Shuffled batches):")
    for idx, batch in enumerate(loader):
        print(f"  Batch {idx}: {batch}")
        
    print("Epoch 2 (Shuffled batches - reset state automatically):")
    for idx, batch in enumerate(loader):
        print(f"  Batch {idx}: {batch}")

    # 2. Generator Pipelines
    raw_text = "Hello, world! Welcome to the world of deep learning and Transformers."
    print(f"\nRaw text: '{raw_text}'")
    
    print("\nProcessing text via Generator Pipeline (Tokens -> Cleaned -> Bigrams):")
    # Define pipeline
    tokens = stream_tokens(raw_text)
    cleaned_tokens = clean_filter(tokens)
    word_bigrams = bigrams(cleaned_tokens)
    
    # Run pipeline lazily
    for bg in word_bigrams:
        print(f"  Bigram: {bg}")

    # or uncomment the following lines to manually call via next() one by one, note that the function pauses at yield
    # and resumes from where it left off when next() is called again
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
    # print(f"  Bigram: {next(word_bigrams)}")
