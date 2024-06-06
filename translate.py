from langdetect import detect
import torch
from transformers import pipeline

import logging
import time


def main():
    # User Message
    input_text = "Qu'est-ce qu'une attaque Drive-by Compromise ? Répondez de façon concrète et concise."
    # Get the language of the input text
    lang = detect(input_text)
    print(f"Detected language: {lang}")
    # Define variables
    model_id = 'Helsinki-NLP/opus-mt-fr-en' if lang == 'fr' else 'Helsinki-NLP/opus-mt-en-fr'
    cache_dir = "./models"

    logging.basicConfig(level=logging.ERROR)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    import os
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    print(f"using {device}")
    pipe = pipeline('translation', model=model_id, tokenizer=model_id, device=device)

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
