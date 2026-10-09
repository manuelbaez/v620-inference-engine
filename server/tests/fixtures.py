"""Test fixtures that need no checkpoint and no GPU: a tiny model directory (a byte-level BPE
tokenizer with the chat special tokens and a minimal chat template, what qwserve.prompt.ChatPrompt
loads) and a FakeEngine for the Scheduler."""

import os
import random
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

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
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([words(40, seed=i) for i in range(2000)], trainer)
    tok.save(os.path.join(path, "tokenizer.json"))
    with open(os.path.join(path, "chat_template.jinja"), "w") as f:
        f.write(TEMPLATE)
    return path


class FakeEngine:
    """Enough of qwserve.engine.Engine for the real Scheduler thread: slots, instant prefill and one
    token per request per step. Failures are injected: `poison` (first token of a prompt whose acquire
    raises, as media that do not match their prompt do), `generate_error` (raised by every step),
    `generate_block` (an Event the step waits on, a stuck GPU), `failed_with` (what failure() reports,
    a rank that failed)."""
    has_mtp = False
    prefill_chunk = 4096

    def __init__(self, capacity=(8192, 8192), step_s=0.0):
        self.capacity = list(capacity)
        self.max_tokens = max(self.capacity)
        self.vram_tokens = list(capacity)
        self.busy = [False] * len(self.capacity)
        self.step_s = step_s
        self.left = {}
        self.steps = 0
        self.poison = set()
        self.generate_error = None
        self.generate_block = None
        self.failed_with = ""

    def cache_stats(self):
        return None

    def slot_timing(self, slot):
        return {"restore_s": 0.001, "prefill_s": 0.002, "save_s": 0.0005, "pin_wait_s": 0.0}

    def slot_len(self, slot):
        return 0

    def moe_use(self):
        return [(0, 0)] * 4

    def failure(self):
        return self.failed_with

    def acquire(self, tokens, max_new, media=None):
        if tokens and tokens[0] in self.poison:
            raise ValueError("media do not match the prompt's placeholders")
        fits = [i for i, c in enumerate(self.capacity) if not self.busy[i] and c >= len(tokens) + max_new]
        if not fits:
            return -1
        slot = min(fits, key=lambda i: self.capacity[i])
        self.busy[slot] = True
        return slot

    def release(self, slot):
        self.busy[slot] = False

    def prefetch(self, slot, tokens, media=None):
        return False

    def set_stop_tokens(self, slot, ids):
        pass

    def begin_prompt(self, slot, tokens, media=None):
        self.left[slot] = len(tokens)
        return 0, None

    def prefill_some(self, slot, n):
        return self.prefill_batch([(slot, n)])[0]

    def prefill_batch(self, pairs):
        done = []
        for slot, n in pairs:
            self.left[slot] -= min(n, self.left[slot])
            done.append(self.left[slot] == 0)
        return done

    def top_logprobs_prompt(self, slot, k):
        return None

    def sample_prompt(self, slot, sampling):
        return 5, 0.0

    def top_logprobs(self, row, k):
        return None

    def generate(self, reqs, k):
        if self.generate_block is not None:
            self.generate_block.wait()
        if self.step_s:
            time.sleep(self.step_s)
        if self.generate_error is not None:
            raise self.generate_error
        self.steps += 1
        return [([5], [0.0], 0, False) for _ in reqs]
