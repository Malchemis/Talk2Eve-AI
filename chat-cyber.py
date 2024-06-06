import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time

from utils import load_tokenizer


def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def load_model_on_available_device(model_id, cache_dir, token, num_threads=8):
    # Set the number of threads
    set_num_threads(num_threads)
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available.")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, token=token,
                                                         device_map='auto', torch_dtype=torch.float16)
            device = 'cuda'
        except Exception as e:
            print(f"Error loading model on GPU: {e}")
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, token=token, device_map='cpu')
            device = 'cpu'
    else:
        print("GPU is not available. Using CPU.")
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, token=token, device_map='cpu')
        device = 'cpu'
    return model, device


def main():
    # Define variables
    # model_id = 'segolilylabs/Lily-Cybersecurity-7B-v0.2'
    model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    cache_dir = 'E:/models/'
    input_text = "What is a Drive-by Compromise attack? Answer matter-of-factly and concisely."

    logging.basicConfig(level=logging.ERROR)

    token, tokenizer = load_tokenizer(model_id, cache_dir)

    try:
        print("Loading model...")
        model, device = load_model_on_available_device(model_id, cache_dir, token)
        print(f"Model loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        # Tokenize input text
        print("Tokenizing input text...")
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        # Generate response
        print("Generating response...")
        start = time.time()
        output = model.generate(input_ids, max_length=120, pad_token_id=tokenizer.eos_token_id)
        end = time.time()
        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)
        print(f"Response generated in {end - start:.2f} seconds.")
    except Exception as e:
        print(f"Error during text generation: {e}")


if __name__ == '__main__':
    main()
