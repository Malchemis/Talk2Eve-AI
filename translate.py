from langdetect import detect
import torch
from transformers import pipeline

import logging
import time


def main():
    # User Message
    input_text = "Bonjour Caroline, comment vas-tu?"
    # Get the language of the input text
    lang = detect(input_text)
    print(f"Detected language: {lang}")
    # Define variables
    model_id = 'Helsinki-NLP/opus-mt-fr-en' if lang == 'fr' else 'Helsinki-NLP/opus-mt-en-fr'
    cache_dir = './models/'

    logging.basicConfig(level=logging.ERROR)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")
    pipe = pipeline('translation', model=model_id, tokenizer=model_id, device=device, cache_dir=cache_dir)

    try:
        # Translate the input text
        print("Translating input text...")
        start = time.time()
        output = pipe(input_text)
        end = time.time()
        # Print the translated text
        print(output[0]['translation_text'])
        print(f"Translation generated in {end - start:.2f} seconds.")
    except Exception as e:
        print(f"Error during translation: {e}")


if __name__ == '__main__':
    main()
