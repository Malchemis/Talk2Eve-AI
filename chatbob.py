import torch
from transformers import pipeline
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig)
from sentence_transformers import SentenceTransformer
from langdetect import detect

import faiss
import pandas as pd

import os

from utils import MyTextStreamer, handle_chat_history

lang_code_to_name = {
    'en': 'English',
    'fr': 'French',
}


class ChatHandler:
    def __init__(self, cache_dir, data_dir, chat_template_path, hub_token, queue, logger):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.chat_template_path = chat_template_path
        self.hub_token = hub_token
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.debug(f"Using device: {self.device}")
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir

        # Load data and models : MITRE database
        self.sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                                              cache_folder=self.cache_dir, token=self.hub_token)
        self.index = faiss.read_index(f'{self.data_dir}/MITRE/mitre.index')
        self.mitre_data = pd.read_csv(f'{self.data_dir}/MITRE/mitre.csv')

        # Load translation pipelines. To translate input prompt for sentence similarity
        self.translation_pipelines = {
            'en-fr': pipeline('translation', model='Helsinki-NLP/opus-mt-en-fr', device=self.device),
            'fr-en': pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en', device=self.device)
        }
        # We use Vigogne as main model
        model_name_or_path, revision = "bofenghuang/vigogne-2-7b-chat", "v2.0"
        self.chat_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision, padding_side="right",
                                                            cache_dir=self.cache_dir, use_fast=False,
                                                            token=self.hub_token)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.chat_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision,
                                                               torch_dtype=torch.float16, device_map="auto",
                                                               cache_dir=self.cache_dir, token=self.hub_token,
                                                               quantization_config=bnb_config)
        self.chat_streamer = MyTextStreamer(self.chat_tokenizer, queue, skip_prompt=True, logger=self.logger)

    def translate_text(self, text, target_lang='en'):
        detected_lang = detect(text)
        if detected_lang not in ['en', 'fr']:
            return text
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
            result = ('\n' +
                      self.mitre_data.iloc[indices[0][i]]["name"] + ': ' +
                      self.mitre_data.iloc[indices[0][i]]["text"])
            results.append(result)
        return results

    # Step 3: Generate chat response
    def get_chat_template(self, user_prompt, context, history):
        with open(self.chat_template_path, 'r') as f:
            default_system_prompt = f.read()
        return (f"{default_system_prompt}\n\n"
                f"user : \n"
                f"- Historique de conversation : \n{handle_chat_history(history)}\n"
                f"- Contexte : {context}\n"
                f"- Question : {user_prompt}\n"
                f"assistant : \n"
                )

    def generate_chat_response(self, chat_history, context, top_p=0.9, top_k=50,
                               repetition_penalty=1.0, max_new_tokens=512, **kwargs):
        input_text = self.get_chat_template(chat_history[-1]['content'], context, chat_history[:-1])
        input_ids = self.chat_tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
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

        chat_history.append({"role": "assistant", "content": generated_text})
        return generated_text, chat_history

    def chat(self, chat_history, last_socket_id):
        self.chat_streamer.last_socket_id = last_socket_id
        self.chat_streamer.first_word = True
        self.logger.debug(f"Chat history:\n{chat_history}")
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
        context = " ".join(search_results) if search_results else ""
        generated_text, chat_history = self.generate_chat_response(chat_history, context)
        self.logger.debug(f"Generated response: {generated_text}")
        return chat_history
