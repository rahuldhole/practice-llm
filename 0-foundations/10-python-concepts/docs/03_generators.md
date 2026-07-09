# 🧬 Tutorial 03: Generators & Iterators

When dealing with large datasets or token streams in machine learning, loading everything into memory at once is often impossible. Python's **Iterators** and **Generators** provide a memory-efficient solution using **lazy evaluation**: elements are generated one at a time, only when requested.

---

## 1. The Iterator Protocol
An iterable is any object you can loop over (e.g. using `for x in obj`). An iterator is the object that actually does the traversal.
For an object to be an iterator, it must implement the **Iterator Protocol** consisting of two methods:
1. `__iter__(self)`: Returns the iterator object itself.
2. `__next__(self)`: Returns the next item from the container. If no more elements remain, it must raise a `StopIteration` exception.

*Code reference*: [`BatchLoader` in generators.py](../src/generators.py#L3-L40)

---

## 2. Generator Functions (`yield`)
Writing custom iterators with classes can be verbose. **Generators** simplify this by using the `yield` keyword.
When a generator function is called, it does not execute the function body. Instead, it returns a generator object. When `next()` is called on this generator, execution runs until it encounters a `yield` statement, returns the yielded value, and pauses state.

**TLDR:** It stop function exection wherever `yield` is written and resumes when `next()` is called. `yield` also returns value like `return`
```python
def simple_generator():
    yield 1
    yield 2
```

*Code reference*: [`stream_tokens` in generators.py](../src/generators.py#L45-L60)

---

## 3. Generator Pipelines
Generators can be chained together to form processing pipelines. Each generator reads elements from the previous generator, processes them, and yields them down the chain. This keeps the memory overhead at $O(1)$ since elements flow through the pipeline one by one.

```text
[Raw Text] --> stream_tokens() --> clean_filter() --> bigrams() --> [Output]
```

*Code reference*: [`clean_filter` and `bigrams` in generators.py](../src/generators.py#L65-L84)

---

## 💡 Practical Challenge
Open [generators.py](../src/generators.py) and execute it with `task run -- src/generators.py`. Observe how data streams through the pipeline. Try creating a pipeline stage `stop_words_remover` that filters out common words (e.g., "to", "the", "of") before generating bigrams.
