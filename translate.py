from langdetect import detect
import torch
from transformers import pipeline

import os
import logging
import time


def main():
    input_text = "Qu'est-ce qu'une attaque Drive-by Compromise ? Répondez de façon concrète et concise."

    lang = detect(input_text)
    print(f"Detected language: {lang}")

    model_id = 'Helsinki-NLP/opus-mt-fr-en' if lang == 'fr' else 'Helsinki-NLP/opus-mt-en-fr'
    cache_dir = "./models"

    logging.basicConfig(level=logging.ERROR)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    print("using ", device := torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    pipe = pipeline('translation', model=model_id, tokenizer=model_id, device=device)
    try:
        print("Translating input text...")
        start = time.time()
        output = pipe(input_text)
        end = time.time()

        print(output[0]['translation_text'])
        print(f"Translation generated in {end - start:.2f} seconds.")
    except Exception as e:
        print(f"Error during translation: {e}")


if __name__ == '__main__':
    main()
