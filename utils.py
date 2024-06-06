from transformers import AutoTokenizer


def load_tokenizer(model_dir, cache_dir):
    try:
        with open('hub-token', 'r') as file:
            token = file.read().strip()
    except FileNotFoundError:
        print("Error: 'hub-token' file not found.")
        return

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir, token=token)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    return token, tokenizer
