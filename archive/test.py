import requests

if __name__ == '__main__':
    url = 'http://localhost:5000/generate'
    payload = {
        "prompt": "As a data scientist, can you explain the concept of regularization in machine learning?",
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

    response = requests.post(url, json=payload)
    print(response.json())
