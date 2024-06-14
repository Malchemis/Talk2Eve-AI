import gc
import torch
from transformers import TextStreamer, AutoTokenizer
from ai_package_for_com.rabbitmq.rabbitmq_handler import RabbitMQHandler


class MyTextStreamer(TextStreamer):
    """
    A TextStreamer that sends the generated text to a queue instead of printing it to stdout.
    Allows for streaming text generation to a web client, an app, etc.
    """
    def __init__(self, tokenizer: 'AutoTokenizer', queue : 'RabbitMQHandler', skip_prompt: bool = False, logger=None,
                 **decode_kwargs):
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
    """Défines the number of threads for PyTorch and Faiss."""
    # Empty cache, get memory back
    torch.cuda.empty_cache()
    gc.collect()
    # Set num workers
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def print_chat(chat):
    """Prints the chat to stdout."""
    for exchange in chat:
        print(f"{exchange['role']}: {exchange['content']}")


def handle_chat_history(chat_history):
    """Format the chat history for prompting."""
    return "\n".join([f"{msg['role']} : {msg['content']}" for msg in chat_history])
