import torch
from transformers import TextStreamer, AutoTokenizer

import gc


class MyTextStreamer(TextStreamer):
    def __init__(self, tokenizer: 'AutoTokenizer', skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)

    # TODO: FOR B3LIOTT. Change the print statement, send text to the queue
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if '</s>' in text:
            text = text[:text.index('</s>')]
        print(text, flush=True, end="" if not stream_end else None)


def set_num_threads(num_threads):
    # Empty cache, get memory back
    torch.cuda.empty_cache()
    gc.collect()
    # Set num workers
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def print_chat(chat):
    for exchange in chat:
        print(f"{exchange['role']}: {exchange['content']}")
