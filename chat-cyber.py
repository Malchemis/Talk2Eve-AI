import logging
import time

from utils import load_tokenizer, load_model_on_available_device


def main():
    # model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    model_id = 'google/gemma-2b-it'
    cache_dir = '/media/malchemis/CYBERTRON/models'
    input_text = "What is a Drive-by Compromise attack? Respond concretely and concisely."

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
        print("Tokenizing input text...")
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        print("Generating response...")
        start = time.time()
        output = model.generate(input_ids, max_length=120, pad_token_id=tokenizer.eos_token_id)
        end = time.time()

        print(response := tokenizer.decode(output[0], skip_special_tokens=True))
        print(f"Response generated in {end - start:.2f} seconds.")
    except Exception as e:
        print(f"Error during text generation: {e}")


if __name__ == '__main__':
    main()
