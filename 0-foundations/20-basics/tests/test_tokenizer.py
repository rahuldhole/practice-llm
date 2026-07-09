import pytest
from basics.tokenizer import CharTokenizer, BPETokenizer

def test_char_tokenizer():
    text = "Hello, GPT!"
    tok = CharTokenizer(text)
    
    assert tok.vocab_size >= len(set(text))
    ids = tok.encode(text)
    assert len(ids) == len(text)
    
    decoded = tok.decode(ids)
    assert decoded == text

def test_bpe_tokenizer_train_and_encode():
    corpus = "banana bandanna cabana"
    tok = BPETokenizer()
    # Initial alphabet in corpus: ' ', 'a', 'b', 'c', 'd', 'n' -> 6 chars
    tok.train(corpus, target_vocab_size=10)
    
    # Should have merged some common pairs, e.g., 'an' or 'na'
    assert tok.vocab_size == 10
    assert len(tok.merges) == 4
    
    # Test encoding and decoding
    encoded = tok.encode("banana")
    decoded = tok.decode(encoded)
    assert decoded == "banana"
    
    # Verify sequence compression (length of encoded BPE should be less than original char length)
    assert len(encoded) < len("banana")

def test_bpe_tokenizer_empty_fails():
    tok = BPETokenizer()
    with pytest.raises(ValueError):
        tok.encode("hello")
