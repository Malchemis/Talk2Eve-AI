import torch
from transformers import pipeline
from transformers import (AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration,
                          BitsAndBytesConfig, GenerationConfig, TextStreamer)
from sentence_transformers import SentenceTransformer
from langdetect import detect

import faiss
import pandas as pd

import traceback
import logging
import yaml

import time
import os

from utils import set_num_threads
from utils import print_chat

lang_code_to_name = {
    'en': 'English',
    'fr': 'French',
}


class MyTextStreamer(TextStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)

    # TODO: FOR B3LIOTT. Change the print statement, send text to the queue
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if '</s>' in text:
            text = text[:text.index('</s>')]
        print(text, flush=True, end="" if not stream_end else None)


class ChatHandler:
    def __init__(self, cache_dir, data_dir, chat_template_path, hub_token):
        logging.basicConfig(level=logging.INFO)
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.chat_template_path = chat_template_path
        self.hub_token = hub_token
        self.logger = logging.getLogger('ChatHandler')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.debug(f"Using device: {self.device}")

        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir

        # Load data and models for MITRE database
        self.sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                                              cache_folder=self.cache_dir, token=self.hub_token)
        self.index = faiss.read_index(f'{self.data_dir}/MITRE/mitre.index')
        self.mitre_data = pd.read_csv(f'{self.data_dir}/MITRE/mitre.csv')

        # Load translation pipelines
        self.translation_pipelines = {
            'en-fr': pipeline('translation', model='Helsinki-NLP/opus-mt-en-fr', device=self.device),
            'fr-en': pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en', device=self.device)
        }

        # Load summarization model
        self.summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn',
                                                                                cache_dir=self.cache_dir,
                                                                                token=self.hub_token)
        self.summarization_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn',
                                                                     cache_dir=self.cache_dir, token=self.hub_token)
        self.summarizer = pipeline("summarization", model=self.summarization_model,
                                   tokenizer=self.summarization_tokenizer)

        # Use vigogne instead of google/gemma-2b-it
        model_name_or_path = "bofenghuang/vigogne-2-7b-chat"
        revision = "v2.0"
        self.chat_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision, padding_side="right",
                                                            use_fast=False, cache_dir=self.cache_dir,
                                                            token=self.hub_token)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.chat_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision,
                                                               torch_dtype=torch.float16, device_map="auto",
                                                               cache_dir=self.cache_dir, token=self.hub_token,
                                                               quantization_config=bnb_config)
        self.chat_streamer = MyTextStreamer(self.chat_tokenizer, skip_prompt=True)

    def translate_text(self, text, target_lang='en'):
        detected_lang = detect(text)
        if detected_lang not in ['en', 'fr']:
            raise ValueError(f"Unsupported language: {detected_lang}")
        if detected_lang == target_lang:
            return text
        self.logger.debug(f"Translating text from {detected_lang} to {target_lang}")
        pipeline_key = f"{detected_lang}-{target_lang}"
        return self.translation_pipelines[pipeline_key](text, max_length=1024)[0]['translation_text']

    # Step 2: Embed query
    def get_query_embedding(self, query):
        return self.sentence_transformer_model.encode([query], convert_to_numpy=True).reshape(1, -1)

    def search_faiss_index(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding, k=k)
        self.logger.info(f"Distances: {distances}")
        self.logger.info(f"Indices: {indices}")
        results = []
        for i in range(k):
            if distances[0][i] > 1.5:
                break
            result = self.mitre_data.iloc[indices[0][i]]["name"] + ': ' + self.mitre_data.iloc[indices[0][i]]["text"]
            results.append(result)
        return results

    # Step 3: Summarize context
    def chunk_text(self, text, sliding_window=256, overlap=64):
        tokens = self.summarization_tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + sliding_window, len(tokens))
            chunks.append(self.summarization_tokenizer.decode(tokens[start:end], skip_special_tokens=True))
            start += sliding_window - overlap
        return chunks

    def _summarize(self, text, max_length=256, min_length=64):
        return self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def summarize(self, text, max_length=512, min_length=256):
        chunks = self.chunk_text(text)
        summaries = [self._summarize(chunk) for chunk in chunks]
        summary = ' '.join(summaries)
        # post process the summary by re-feeding it to the summarizer
        return self._summarize(summary, max_length, min_length)

    # Step 4: Generate chat response
    def generate_chat_response(self, chat_history, context, top_p=0.9, top_k=50,
                               repetition_penalty=1.0, max_new_tokens=512, **kwargs):
        conversation = chat_history.copy()
        conversation[-1]['content'] += f"\nContext: {context}\nAlways answer in French. Répond toujours en français."
        input_ids = self.chat_tokenizer.apply_chat_template(conversation, add_generation_prompt=True,
                                                            return_tensors="pt").to(self.chat_model.device)
        input_length = input_ids.shape[1]

        generated_outputs = self.chat_model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.chat_tokenizer.pad_token_id,
                eos_token_id=self.chat_tokenizer.eos_token_id,
                **kwargs,
            ),
            streamer=self.chat_streamer,
            return_dict_in_generate=True,
        )

        generated_tokens = generated_outputs.sequences[0, input_length:]
        generated_text = self.chat_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def handle_user_query(self, chat_history):
        # Step 1: Translate query to English if necessary
        translated_query = self.translate_text(chat_history[-1]['content'], target_lang='en')
        self.logger.debug(f"Translated query: {translated_query}")

        # Step 2: Embed query
        query_embedding = self.get_query_embedding(translated_query)
        # self.logger.debug(f"Query embedding: {query_embedding}")

        # Step 3: Search FAISS index
        search_results = self.search_faiss_index(query_embedding)
        self.logger.debug(f"Search results: {search_results}")

        # Step 4: Generate chat response
        context = " ".join(search_results) if search_results else 'None'
        response = self.generate_chat_response(chat_history, context)
        self.logger.debug(f"Generated response: {response}")
        return response


def main(config):
    set_num_threads(config['torch']['num_workers'])
    chat_handler = ChatHandler(**config['chat_handler'])
    chat_history = []
    while True:
        user_prompt = input("Enter your query: ")
        if user_prompt == 'exit':
            break
        start = time.time()
        try:
            chat_history.append({'role': 'user', 'content': user_prompt})
            model_response = chat_handler.handle_user_query(chat_history)
        except:
            traceback.print_exc()
            continue
        chat_history.append({'role': 'assistant', 'content': model_response})
        print_chat(chat_history)
        print('\nObtained in', time.time() - start, 'seconds')


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
    main(config_file)
