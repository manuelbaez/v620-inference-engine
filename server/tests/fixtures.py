"""A tiny model directory for tests that need no checkpoint: a byte-level BPE tokenizer with the
chat special tokens and a minimal chat template (what qwserve.prompt.ChatPrompt loads)."""

import os
import random

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
           "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
TEMPLATE = ("{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
            "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- if enable_thinking %}<think>\n{%- endif %}"
            "{%- endif %}")
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon phi chi psi omega function parameter value return const static void int float struct class "
         "import export while for if else switch case break continue true false null none").split()


def words(n, seed=1):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) + (str(rng.randrange(100)) if rng.random() < 0.3 else "") for _ in range(n))


def make_model_dir(path, vocab_size=2000):
    """Writes tokenizer.json and chat_template.jinja into `path` and returns it."""
    os.makedirs(path, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([words(40, seed=i) for i in range(2000)], trainer)
    tok.save(os.path.join(path, "tokenizer.json"))
    with open(os.path.join(path, "chat_template.jinja"), "w") as f:
        f.write(TEMPLATE)
    return path
