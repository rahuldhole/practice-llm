import os

class SimpleTokenizer:
    def __init__(self, vocab_size=2000):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.pad_token_id = 0
        self.unk_token_id = 1
        self._vocab_size_target = vocab_size

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def fit(self, text_or_file):
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=self._vocab_size_target)
        if os.path.exists(text_or_file):
            self.tokenizer.train([text_or_file], trainer=trainer)
        else:
            self.tokenizer.train_from_iterator([text_or_file], trainer=trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path):
        from tokenizers import Tokenizer
        obj = cls()
        obj.tokenizer = Tokenizer.from_file(path)
        return obj

if __name__ == "__main__":
    tokenizer = SimpleTokenizer(vocab_size=100)
    test_text = "Hello world. Hello AI. Python is fun. The sky is blue."
    tokenizer.fit(test_text)
    print("Vocab size:", tokenizer.vocab_size)
    encoded = tokenizer.encode("Hello world.")
    print("Encoded 'Hello world.':", encoded)
    print("Decoded:", tokenizer.decode(encoded))
