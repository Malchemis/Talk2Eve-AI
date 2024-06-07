# Description: This script is used to generate responses from a Large Langage Model like Gemma or Mistral AI.
from utils import load_tokenizer, load_model_on_available_device
import time


def chat(model_id: str, cache_dir: str, input_text: str, add_entries: str) -> str:
    token, tokenizer = load_tokenizer(model_id, cache_dir)

    try:
        model, device = load_model_on_available_device(model_id, cache_dir, token)
    except Exception as e:
        print(f"Error loading model: {e}")
        return ''

    try:
        prompt = get_chat_template(input_text, add_entries)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=1024)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during text generation: {e}")
        return ''


default_system_prompt = r"""
<start_of_turn>user
You are a helpful, respectful, and honest assistant.
Always answer as helpfully as possible using the context text provided.
Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make sense or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information. Just say I don't know.

You will always be given some context from the MITRE Database to generate a response. You may also be given the history
of the conversation to help.

BEGIN EXAMPLE
Context: MITRE ATT&CK (Adversarial Tactics, Techniques, and Common Knowledge) is a globally-accessible knowledge base of adversary tactics and techniques based on real-world observations.
Question: What is MITRE ATT&CK?
Answer: 
MITRE ATT&CK (Adversarial Tactics, Techniques, and Common Knowledge) is a comprehensive framework developed by the MITRE Corporation that details the behaviors and methods used by cyber attackers based on real-world observations. It is structured around three main components:

1. Tactics: The objectives of an attack (e.g., persistence, data exfiltration).
2. Techniques and Sub-Techniques: The methods used to achieve these objectives.
3. Procedures: Specific implementations of these methods.

The framework is organized into matrices tailored for different environments, including enterprise (Windows, macOS, Linux, cloud, containers), mobile, and industrial control systems (ICS). It serves multiple use cases such as enhancing threat intelligence, aiding security operations, and supporting red and blue teaming exercises. MITRE ATT&CK is community-driven, continuously updated, and widely used to improve cybersecurity defenses.
END EXAMPLE

"""


def get_chat_template(prompt: str, add_entries):
    return (f"{default_system_prompt}\n\nContext: {add_entries}\nQuestion: {prompt}"
            f"<end_of_turn>\n<start_of_turn>Answer: ")


if __name__ == '__main__':
    # model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    model_name = 'google/gemma-2b-it'
    cache_folder = '/media/malchemis/CYBERTRON/models'
    user_prompt = "What is a Drive-by Compromise attack?" # In-context user
    # user_prompt = "How to cook a pizza?"  # Out-of-context user
    with open('data/MITRE/Drive-by-Compromise.txt', 'r') as f:
        mitre_entries = f.read()
    start = time.time()
    print(chat(model_name, cache_folder, user_prompt, mitre_entries[:100]))
    print(f"Executed in {time.time()-start}s")
