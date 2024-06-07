# Description: Chatbot pipeline using SentenceTransformers and FAISS for real-time query handling.
from sentence_transformers import SentenceTransformer
import faiss

from summarizer import summarize
from translate import translate
from cyberLLM import chat

import os
import logging
import time

# Load pre-trained model
cache_dir = '/media/malchemis/CYBERTRON/models'
data_dir = '/media/malchemis/CYBERTRON/data'
sentenceTransformerModel = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder=cache_dir)

# Pre-compute embeddings for MITRE techniques and tactics
mitre_entries, mitre_titles = [], []
for entry in os.listdir(f'{data_dir}/MITRE'):
    if entry.endswith('.txt'):
        with open(f'{data_dir}/MITRE/{entry}', 'r') as f:
            mitre_entries.append(f.read())
            mitre_titles.append(entry.split('.')[0])
mitre_embeddings = sentenceTransformerModel.encode(mitre_entries, convert_to_tensor=True)

# Index embeddings using FAISS
index = faiss.IndexFlatL2(mitre_embeddings.shape[1])
index.add(x=mitre_embeddings.cpu().numpy())


# Real-time query handling
def handle_user_query(query, logger=logging.getLogger()):
    # Translate query if needed
    query = translate(query, model_id='Helsinki-NLP/opus-mt-fr-en', cache_dir=cache_dir, output_lang='en')
    logger.info(f"Translated query: {query}")

    # Vectorize query
    query_embedding = sentenceTransformerModel.encode([query], convert_to_tensor=True)
    logger.info(f"Query embedding shape: {query_embedding.shape}")

    # Search in vector database
    D, I = index.search(x=query_embedding.cpu().numpy(), k=1)
    logger.info(f"Search results: {I}")
    logger.info(f"Distances: {D}")

    # Retrieve and combine information from top results
    top_results = [mitre_titles[i] + ' ' + summarize(mitre_entries[i], max_length=512) for i in I[0]]
    # top_results = [mitre_entries[i] for i in I[0]]
    logger.info(f"Top results (summarized): {top_results}")

    # Generate a precise answer using the chatbot
    context = " ".join(top_results)  # Combine top results
    answer = chat(model_id='google/gemma-2b-it', cache_dir=cache_dir, input_text=query, add_entries=context)
    logger.info(f"Generated output: {answer}")
    answer = answer.split('Answer: ')[-1].strip()
    logger.info(f"Final answer (en): {answer}")

    # Translate answer if needed
    translated = translate(answer, model_id='Helsinki-NLP/opus-mt-en-fr', cache_dir=cache_dir, output_lang='fr')
    return translated


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    user_prompt = "Explique ce qu'est un Buffer Overflow ?"
    # user_prompt = "Qu'est-ce qu'une attaque Drive-by Compromise ?"  # In-context user French
    # user_prompt = "What is a Drive-by Compromise attack?"  # In-context user English
    # user_prompt = "Comment cuisiner une pizza ?"  # Out-of-context user French
    # user_prompt = "How to cook a pizza?"  # Out-of-context user English
    response = handle_user_query(user_prompt)
    print(response)
    print(f"Execution time: {time.time() - start:.2f} seconds")
