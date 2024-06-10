"""
Takes as input a file containing a list of strings and computes the similarity between the first sentence and all the
other sentences.
The similarity is computed using the Cosine similarity coefficient.
The output is stored in a file, where each line contains the sentence, the similarity score ordered in descending order.
"""

# Modules
import sys  # to read arguments
from pathlib import Path  # to handle files

# Model and tokenizer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def main():
    # Read arguments
    if len(sys.argv) != 3:
        print("Usage: python similarity.py input_file output_file")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    # Load model and tokenizer
    tokenizer, model = get_model(model_name)

    # Read sentences from file
    sentences = read_file(input_file)

    # Compute similarity
    similarity_scores = []
    for sentence in sentences[1:]:
        similarity_scores.append((sentence, similarity(sentences[0], sentence, tokenizer, model)))

    # Sort by similarity score
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Write to output file
    with open(output_file, 'w') as file:
        for sentence, score in similarity_scores:
            file.write(f'{sentence.strip()}\t{score}\n')


def similarity(sentence1, sentence2, tokenizer, model):
    sentence1 = sentences_to_embeddings([sentence1], tokenizer, model)
    sentence2 = sentences_to_embeddings([sentence2], tokenizer, model)
    return F.cosine_similarity(sentence1, sentence2).item()  # cosine similarity


def sentences_to_embeddings(sentences, tokenizer, model):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


# Utils
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


def get_model(model_name='sentence-transformers/all-MiniLM-L6-v2',
              tokenize_name='sentence-transformers/all-MiniLM-L6-v2',
              output_path='E:/models/'):
    tokenizer = AutoTokenizer.from_pretrained(tokenize_name)
    model = AutoModel.from_pretrained(model_name, cache_dir=output_path)
    return tokenizer, model


if __name__ == '__main__':
    main()
