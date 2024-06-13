import torch
from transformers import TextStreamer, AutoTokenizer

import gc


class MyTextStreamer(TextStreamer):
    def __init__(self, tokenizer: 'AutoTokenizer', queue, skip_prompt: bool = False, logger=None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.first_word = True
        self.queue = queue
        self.last_socket_id = None
        self.access_token = None
        self.logger = logger

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if '<s>' in text:
            text = text[text.index('<s>') + 3:]
        if '</s>' in text:
            text = text[:text.index('</s>')]
        if self.first_word:
            category = "message"
            self.first_word = False
        else:
            category = "word"
        response = {"status": category, "socket_id": self.last_socket_id, 'access_token': self.access_token,
                    category: text}
        self.logger.debug(f"Sending {text} to the queue")
        self.queue.send_result(response)
        # print(text, flush=True, end="" if not stream_end else None)


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


def handle_chat_history(chat_history):
    return "\n".join([f"{msg['role']} : {msg['content']}" for msg in chat_history])
