import pytest
from python_concepts.generators import BatchLoader, stream_tokens, clean_filter, bigrams

def test_batch_loader():
    data = list(range(1, 10))  # 1 to 9
    loader = BatchLoader(data, batch_size=4, shuffle=False)
    
    # Verify iterator protocol
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0] == [1, 2, 3, 4]
    assert batches[1] == [5, 6, 7, 8]
    assert batches[2] == [9]
    
    # Test restartability
    batches_again = list(loader)
    assert len(batches_again) == 3
    assert batches_again[0] == [1, 2, 3, 4]


def test_batch_loader_shuffle():
    data = list(range(100))
    loader = BatchLoader(data, batch_size=10, shuffle=True)
    batches = list(loader)
    
    # Check that it returned all elements, but the order should be shuffled
    flattened = [x for b in batches for x in b]
    assert sorted(flattened) == data
    assert flattened != data  # highly unlikely to match by random chance


def test_generator_pipeline():
    text = "Hello, testing! This is an iterative pipeline."
    
    tokens = stream_tokens(text)
    cleaned = clean_filter(tokens)
    bgs = list(bigrams(cleaned))
    
    # Expected cleaned: ["hello", "testing", "this", "is", "an", "iterative", "pipeline"]
    # Expected bigrams:
    # ("hello", "testing"), ("testing", "this"), ("this", "is"), ("is", "an"), ("an", "iterative"), ("iterative", "pipeline")
    
    assert len(bgs) == 6
    assert bgs[0] == ("hello", "testing")
    assert bgs[-1] == ("iterative", "pipeline")
