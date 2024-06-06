# Description: This script is used to generate responses from a Large Langage Model like Gemma or Mistral AI.
from utils import load_tokenizer, load_model_on_available_device


def chat(model_id: str, cache_dir: str, input_text: str) -> str:
    token, tokenizer = load_tokenizer(model_id, cache_dir)

    try:
        model, device = load_model_on_available_device(model_id, cache_dir, token)
    except Exception as e:
        print(f"Error loading model: {e}")
        return ''

    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        output = model.generate(input_ids, max_length=len(input_text)+500, pad_token_id=tokenizer.eos_token_id)

        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during text generation: {e}")
        return ''


if __name__ == '__main__':
    # model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    model_name = 'google/gemma-2b-it'
    cache_folder = '/media/malchemis/CYBERTRON/models'
    prompt = "What is a Drive-by Compromise attack?"
    chat(model_name, cache_folder, prompt)
