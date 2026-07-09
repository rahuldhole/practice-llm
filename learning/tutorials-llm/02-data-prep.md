# 2. Data Preparation: Turning Words into Numbers

**TLDR:** Preparing and cleaning datasets for model training.

<details>
<summary>Why do we need to prepare data?</summary>

Computers are super calculators. They only understand **numbers**, not English letters or words!

Before our AI brain can learn from books, we have to translate all the words into lists of numbers. This translation process is called **Tokenization**. 

Imagine giving every word in the dictionary its own secret ID badge number.

</details>

## Tokenization Flow

```mermaid
flowchart TD
    A["I love coding!"] -->|1. Clean Text| B["i love coding !"]
    B -->|2. Split| C["['i', 'love', 'coding', '!']"]
    C -->|3. Lookup IDs| D["[42, 108, 903, 5]"]
```

### The Preparation Steps

| Step | Action | Kid-friendly Analogy |
|---|---|---|
| **1. Collect** | Gather a bunch of text. | Going to the library and grabbing a huge stack of books. |
| **2. Clean** | Remove weird symbols or messy text. | Wiping the dirt off old book covers so they are easy to read. |
| **3. Tokenize** | Split text into pieces (tokens). | Chopping a chocolate bar into tiny squares. |
| **4. Encode** | Give each piece a number ID. | Giving each square a barcode. |

<details>
<summary>💻 See the Code (How we do it in our LLM)</summary>

In our `llm/tokenizer.py`, we build our own dictionary and assign ID numbers using **BPE** (Byte-Pair Encoding), the same way big models do it!

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# 1. Create a blank dictionary
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 2. Train it on our books to find common word pieces
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=2000)
tokenizer.train(["my_books.txt"], trainer=trainer)

# 3. Turn words into numbers!
ids = tokenizer.encode("I love coding!").ids
# Result: [42, 108, 903, 5]
```

</details>

<details>
<summary>Scaling up to SOTA (State of the Art)</summary>

When building models like GPT-4 or LLaMA, this step is basically the exact same! 
The only difference is that instead of using one little book, they download the entire internet (petabytes of text!) and use a very smart dictionary (like Byte-Pair Encoding) to chop up words more efficiently.

</details>

## 📚 Resources for Deep Learning
- [HuggingFace Tokenizers Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Byte Pair Encoding (BPE) explanation](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- [The Illustrated Word2vec (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/)
