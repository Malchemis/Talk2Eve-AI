import logging
import time

from utils import load_tokenizer, load_model_on_available_device


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
