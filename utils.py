from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_tokenizer(model_dir, cache_dir):
    try:
        with open('hub-token', 'r') as file:
            token = file.read().strip()
    except FileNotFoundError:
        print("Error: 'hub-token' file not found.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir, token=token)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    return token, tokenizer


def load_model_on_available_device(model_id, cache_dir, token, num_threads=8):
    # Set the number of threads
    set_num_threads(num_threads)

    # Check if GPU is available
    if torch.cuda.is_available():
        # device_map = {0: 'cuda', 1: 'cpu'}
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, token=token,
                                                         device_map='auto', torch_dtype=torch.float16, )
            # attn_implementation="flash_attention_2")
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


def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
