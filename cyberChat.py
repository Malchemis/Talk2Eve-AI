# Description: Chatbot pipeline using SentenceTransformers and FAISS for real-time query handling.
from sentence_transformers import SentenceTransformer
import faiss

from summarizer import summarize
from translate import translate
from cyberLLM import chat

import logging
import time
import pandas as pd

# Load pre-trained model
cache_dir = '/media/basic/CYBERTRON/models'
data_dir = '/media/basic/CYBERTRON/data'
sentenceTransformerModel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir)

# Load MITRE data
mitre = pd.read_csv(f'{data_dir}/MITRE/mitre.csv')

# Load the FAISS index
index = faiss.read_index('/media/basic/CYBERTRON/data/MITRE/mitre.index')


# Real-time query handling
def handle_user_query(query, logger=logging.getLogger()):
    # Translate query if needed
    query = translate(query, model_id='Helsinki-NLP/opus-mt-fr-en', cache_dir=cache_dir, output_lang='en')
    logger.info(f"Translated query: {query}")

    # Vectorize query
    query_embedding = sentenceTransformerModel.encode([query], convert_to_numpy=True).reshape(1, -1)
    logger.info(f"Query embedding shape: {query_embedding.shape}")

    # Search in vector database
    k = 1  # Number of results to retrieve
    distances, indices = index.search(x=query_embedding, k=k)
    logger.info(f"Search results: {indices}")
    logger.info(f"Distances: {distances}")

    # Retrieve and combine information from top results
    top_results = []
    for i in range(k):
        top_results.append(mitre.iloc[indices[0][i]]["name"] + ': ' + summarize(mitre.iloc[indices[0][i]]["text"]))
    logger.info(f"Top results (summarized): {top_results}")

    # Generate a precise answer using the chatbot
    context = " ".join(top_results)  # Combine top results
    answer = chat(model_id='google/gemma-2b-it', cache_dir=cache_dir, input_text=query, add_entries=context)
    logger.info(f"Generated output: {answer}")
    answer = answer.split('Answer: ')[-1].strip()

    # Translate answer if needed
    translated = translate(answer, model_id='Helsinki-NLP/opus-mt-en-fr', cache_dir=cache_dir, output_lang='fr')
    return translated


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    # user_prompt = "Explique ce qu'est un Buffer Overflow ?"
    user_prompt = "Qu'est-ce qu'une attaque Drive-by Compromise ?"  # In-context user French
    # user_prompt = "What is a Drive-by Compromise attack?"  # In-context user English
    # user_prompt = "Comment cuisiner une pizza ?"  # Out-of-context user French
    # user_prompt = "How to cook a pizza?"  # Out-of-context user English
    response = handle_user_query(user_prompt)
    print(response)
    print(f"Execution time: {time.time() - start:.2f} seconds")
