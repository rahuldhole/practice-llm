class CharTokenizer:
    """
    A simple character-level tokenizer.
    Maps each character to a unique integer ID.
    """
    def __init__(self, text=""):
        if text:
            self.chars = sorted(list(set(text)))
        else:
            # Default fallback characters
            self.chars = sorted(list(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\n")))
        
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi.get(c, self.stoi.get(' ', 0)) for c in text]

    def decode(self, ids):
        return ''.join([self.itos.get(i, '?') for i in ids])


class BPETokenizer:
    """
    A minimal, educational Byte Pair Encoding (BPE) tokenizer implemented from scratch.
    """
    def __init__(self):
        self.vocab = {}
        self.merges = {}  # tuple (id1, id2) -> merged_id
        self.vocab_size = 0

    def _get_stats(self, ids):
        """
        Count occurrences of all adjacent pairs of tokens in ids.
        """
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        """
        Replace all occurrences of 'pair' with new token 'idx'.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, target_vocab_size):
        """
        Train BPE on text until vocabulary size reaches target_vocab_size.
        """
        # Start with individual characters (Unicode code points or character symbols)
        raw_chars = sorted(list(set(text)))
        self.vocab = {i: c.encode('utf-8') if isinstance(c, str) else bytes([c]) for i, c in enumerate(raw_chars)}
        
        # Initial token sequence is character indices
        char_to_id = {c: i for i, c in enumerate(raw_chars)}
        ids = [char_to_id[c] for c in text]
        
        num_merges = target_vocab_size - len(raw_chars)
        current_vocab_size = len(raw_chars)
        
        # Keep track of merges
        self.merges = {}
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            new_id = current_vocab_size + i
            
            # Merge this pair in our sequence
            ids = self._merge(ids, best_pair, new_id)
            
            # Save the merge rule and update vocabulary
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        """
        Encode text into BPE token IDs by greedily applying merges in training order.
        """
        if not self.vocab:
            raise ValueError("Tokenizer has not been trained yet.")
            
        # Initialize sequence as individual characters (Unicode characters mapping to original vocab)
        # First we need to map characters to their base IDs
        char_to_id = {val.decode('utf-8', errors='replace'): key for key, val in self.vocab.items() if len(val.decode('utf-8', errors='replace')) == 1}
        
        # If text contains characters not in vocab, map them to a fallback character or ignore
        ids = []
        for c in text:
            if c in char_to_id:
                ids.append(char_to_id[c])
            else:
                # fall back to character code point if possible, or skip
                ids.append(0) 

        # Greedily apply merges in the exact order they were learned during training
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find which of the available pairs occurred first in the merges dictionary
            # The smaller the merge value in self.merges, the higher the priority
            pair_to_merge = None
            min_merge_idx = float('inf')
            
            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_merge_idx:
                        min_merge_idx = self.merges[pair]
                        pair_to_merge = pair
            
            if pair_to_merge is None:
                break # No more merges can be applied
                
            ids = self._merge(ids, pair_to_merge, self.merges[pair_to_merge])
            
        return ids

    def decode(self, ids):
        """
        Decode BPE token IDs back into string.
        """
        tokens = []
        for idx in ids:
            if idx in self.vocab:
                tokens.append(self.vocab[idx])
            else:
                tokens.append(b'?')
        # Reconstruct string by joining bytes
        return b''.join(tokens).decode('utf-8', errors='replace')


if __name__ == "__main__":
    print("--- Character Tokenizer Demo ---")
    corpus = "hello world! hello from scratch transformer!"
    char_tok = CharTokenizer(corpus)
    print(f"Vocab size: {char_tok.vocab_size}")
    ids = char_tok.encode("hello scratch")
    print(f"Encoded 'hello scratch': {ids}")
    print(f"Decoded: '{char_tok.decode(ids)}'")
    
    print("\n--- BPE Tokenizer Demo ---")
    bpe_tok = BPETokenizer()
    # Train on corpus
    bpe_tok.train(corpus, target_vocab_size=35)
    print(f"BPE Vocab Size: {bpe_tok.vocab_size}")
    print(f"BPE Merges: {bpe_tok.merges}")
    
    bpe_ids = bpe_tok.encode("hello scratch")
    print(f"BPE Encoded: {bpe_ids}")
    print(f"BPE Decoded: '{bpe_tok.decode(bpe_ids)}'")
