from langdetect import detect
import torch
from transformers import pipeline

import os
import logging


def translate(
        input_text="Qu'est-ce qu'une attaque Drive-by Compromise ?",
        model_id='Helsinki-NLP/opus-mt-fr-en',
        cache_dir="./models",
        output_lang='en'
):
    detected_lang = detect(input_text)
    if detected_lang == output_lang:
        return input_text

    logging.basicConfig(level=logging.ERROR)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pipe = pipeline('translation', model=model_id, tokenizer=model_id, device=device)
    try:
        return pipe(input_text)[0]['translation_text']
    except Exception as e:
        print(f"Error during translation: {e}")


if __name__ == '__main__':
    translate(model_id='Helsinki-NLP/opus-mt-en-fr', output_lang='fr')
