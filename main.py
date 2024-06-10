import torch

from transformers import pipeline
from transformers import (AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration)
from sentence_transformers import SentenceTransformer
from langdetect import detect

import faiss
import pandas as pd

import logging
import yaml
import os
import time

from utils import set_num_threads


class ChatHandler:
    def __init__(self, cache_dir, data_dir, chat_template_path, hub_token):
        logging.basicConfig(level=logging.INFO)
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.chat_template_path = chat_template_path
        self.hub_token = hub_token
        self.logger = logging.getLogger('ChatHandler')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.logger.debug(f"Using device: {self.device}")

        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir

        # Load data and models for MITRE database
        self.sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=self.cache_dir, token=self.hub_token)
        self.index = faiss.read_index(f'{self.data_dir}/MITRE/mitre.index')
        self.mitre_data = pd.read_csv(f'{self.data_dir}/MITRE/mitre.csv')

        # Load translation pipelines
        self.translation_pipelines = {
            'en-fr': pipeline('translation', model='Helsinki-NLP/opus-mt-en-fr', device=self.device),
            'fr-en': pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en', device=self.device)
        }
        # self.translation_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", cache_dir=self.cache_dir, token=self.hub_token)
        # self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", cache_dir=self.cache_dir, token=self.hub_token, device_map='auto', torch_dtype=torch.float16)

        # Load summarization model
        self.summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', cache_dir=self.cache_dir, token=self.hub_token)
        self.summarization_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', cache_dir=self.cache_dir, token=self.hub_token)
        self.summarizer = pipeline("summarization", model=self.summarization_model, tokenizer=self.summarization_tokenizer)

        # Load chat model and tokenizer
        self.chat_model_id = 'google/gemma-2b-it'
        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id, cache_dir=self.cache_dir, token=self.hub_token)
        self.chat_model = AutoModelForCausalLM.from_pretrained(self.chat_model_id, cache_dir=self.cache_dir, device_map='auto', torch_dtype=torch.float16, token=self.hub_token)

    # Step 1: Translate query
    def translate_text(self, text, target_lang='en'):
        detected_lang = detect(text)
        if detected_lang not in ['en', 'fr']:
            raise ValueError(f"Unsupported language: {detected_lang}")
        if detected_lang == target_lang:
            return text
        self.logger.debug(f"Translating text from {detected_lang} to {target_lang}")
        pipeline_key = f"{detected_lang}-{target_lang}"
        return self.translation_pipelines[pipeline_key](text, max_length=1024)[0]['translation_text']
    # def translate_text(self, text, target_lang='en'):
    #     detected_lang = detect(text)
    #     if detected_lang == target_lang:
    #         return text
    #     input_text = f"translate {detected_lang} to {target_lang}: {text}"
    #     inputs = self.translation_tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    #     outputs = self.translation_model.generate(inputs.input_ids.to(self.device), max_length=1024)
    #     return self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True).split(': ')[-1].strip()

    # Step 2: Embed query
    def get_query_embedding(self, query):
        return self.sentence_transformer_model.encode([query], convert_to_numpy=True).reshape(1, -1)

    def search_faiss_index(self, query_embedding, k=1):
        distances, indices = self.index.search(query_embedding, k=k)
        self.logger.info(f"Distances: {distances}")
        self.logger.info(f"Indices: {indices}")
        results = []
        for i in range(k):
            if distances[0][i] > 1.3:
                break
            result = self.mitre_data.iloc[indices[0][i]]["name"] + ': ' + self.summarize(self.mitre_data.iloc[indices[0][i]]["text"])
            results.append(result)
        return results

    # Step 3: Summarize context
    def chunk_text(self, text, sliding_window=128, overlap=32):
        tokens = self.summarization_tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + sliding_window, len(tokens))
            chunks.append(self.summarization_tokenizer.decode(tokens[start:end], skip_special_tokens=True))
            start += sliding_window - overlap
        return chunks

    def _summarize(self, text, max_length=128, min_length=64):
        return self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def summarize(self, text, max_length=512, min_length=256):
        chunks = self.chunk_text(text)
        summaries = [self._summarize(chunk) for chunk in chunks]
        summary = ' '.join(summaries)
        # post process the summary by re-feeding it to the summarizer
        return self._summarize(summary, max_length, min_length)

    # Step 4: Generate chat response
    def generate_chat_response(self, prompt, context, history=''):
        input_text = self.get_chat_template(prompt, context, history)
        inputs = self.chat_tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
        outputs = self.chat_model.generate(inputs.to(self.device), max_new_tokens=1024)
        return self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_chat_template(self, user_prompt, context, history=''):
        with open(self.chat_template_path, 'r') as f:
            default_system_prompt = f.read()
        return (f"{default_system_prompt}\n\n"
                f"History: {history if history else 'None'}\n"
                f"Context: {context}\n"
                f"Question: {user_prompt}"
                f"<end_of_turn>\n<start_of_turn>Answer: ")

    # Chat handler
    def handle_history(self, history):
        # Translate history to English
        if history == '':
            return None
        history = self.translate_text(history, target_lang='en')
        return history

    def handle_user_query(self, query, history=''):
        # Step 1: Translate query to English if necessary
        translated_query = self.translate_text(query, target_lang='en')
        self.logger.debug(f"Translated query: {translated_query}")

        # Step 2: Embed query
        query_embedding = self.get_query_embedding(translated_query)
        # self.logger.debug(f"Query embedding: {query_embedding}")

        # Step 3: Search FAISS index
        search_results = self.search_faiss_index(query_embedding)
        self.logger.debug(f"Search results: {search_results}")

        # Step 4: Generate chat response
        context = " ".join(search_results) if search_results else 'None'
        response = self.generate_chat_response(translated_query, context, self.handle_history(history)).split('Answer: ')[-1].strip()
        self.logger.debug(f"Generated response: {response}")

        # Step 5: Translate response back to the original language
        final_response = self.translate_text(response, target_lang='fr')
        return final_response


def main(config):
    set_num_threads(config['torch']['num_workers'])
    chat_handler = ChatHandler(**config['chat_handler'])
    history = ''
    while True:
        user_prompt = input("Enter your query: ")
        if user_prompt == 'exit':
            break
        start = time.time()
        try:
            model_response = chat_handler.handle_user_query(user_prompt, history)
        except Exception as e:
            print(f"Error: {e}")
            continue
        print(model_response, '\nObtained in', time.time()-start, 'seconds')
        history += f"user: {user_prompt}\n"
        history += f"assistant: {model_response}\n"
        print('Chat:\n', history)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
    main(config_file)
