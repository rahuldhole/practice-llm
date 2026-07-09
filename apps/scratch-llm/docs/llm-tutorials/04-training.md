# 4. Training: Teaching the AI

<details>
<summary>How does the AI actually learn?</summary>

The AI starts out completely clueless. If you ask it a question, it just babbles random words!

To teach it, we play a game:
1. We show it a piece of text: "The dog barked at the..."
2. The AI guesses the next word. Let's say it guesses "apple".
3. We tell it: "Wrong! The answer was 'cat'."
4. We adjust the AI's brain slightly so it is more likely to guess "cat" next time.
5. We repeat this millions of times!

</details>

## The Learning Loop

```mermaid
flowchart LR
    A[Show Text: "The dog barked at the"] --> B(Model Guesses)
    B --> C{"Guess: 'apple'?"}
    C -- Wrong! Target: 'cat' --> D[Calculate Error / Loss]
    D --> E[Update Brain / Backpropagation]
    E -.->|Repeat millions of times| A
```

### The Teacher's Tools

| Term | What it means | Kid-friendly Analogy |
|---|---|---|
| **Loss** | A score of how wrong the guess was. We want a low score! | Getting a bad grade on a test. The lower the mistakes, the better. |
| **Optimizer** | The tool that actually twists the knobs in the brain to fix the mistakes. | A tutor who shows you exactly how to fix your homework. |
| **Epoch** | Going through your entire stack of books one full time. | Reading every book in your school library once. |
| **Backpropagation** | Sending the error backwards through the brain to see what went wrong. | Tracing your steps back when you lose your toy to find out where you left it. |

<details>
<summary>Scaling up to SOTA (State of the Art)</summary>

When training massive models like LLaMA or GPT, the loop is identical! 
The only difference is scale. Instead of training on a single laptop for an hour, they train on clusters of thousands of powerful graphics cards (GPUs) for months at a time! 

You can build a small version on your laptop here, and one day take the exact same code and run it on a supercomputer!

</details>
