import re

class SimpleTokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        
        # Add special tokens
        self.pad_token_id = self.add_word("[PAD]")
        self.unk_token_id = self.add_word("[UNK]")

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.vocab_size
            self.id2word[self.vocab_size] = word
            self.vocab_size += 1
        return self.word2id[word]

    def _split(self, text):
        # simple split by whitespace and punctuation
        words = re.findall(r"[\w']+|[.,!?;]", text)
        return words

    def fit(self, text):
        words = self._split(text)
        for word in words:
            self.add_word(word)

    def encode(self, text):
        words = self._split(text)
        return [self.word2id.get(word, self.unk_token_id) for word in words]

    def decode(self, token_ids):
        return " ".join([self.id2word.get(idx, "[UNK]") for idx in token_ids])

if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    with open("dataset.txt", "r") as f:
        text = f.read()
    tokenizer.fit(text)
    print("Vocab size:", tokenizer.vocab_size)
    print("Vocab:", tokenizer.word2id)
    encoded = tokenizer.encode("Hello world.")
    print("Encoded 'Hello world.':", encoded)
    print("Decoded:", tokenizer.decode(encoded))
