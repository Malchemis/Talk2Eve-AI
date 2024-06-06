import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, GenerationConfig
from flask import Flask, request, jsonify

# Disable memory efficient SDP and flash SDP
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Define the model name/path
model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
cache_dir = '../models/'
with open('../hub-token', 'r') as f:
    huggingface_hub_token = f.read()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=huggingface_hub_token)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             cache_dir=cache_dir,
                                             token=huggingface_hub_token
)

streamer = TextStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
tokenizer.chat_template = '''
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% elif true == true %}{% set loop_messages = messages %}
    {% set system_message = 'Vous êtes un Assistant du site de l'Université Polytechnique Hauts-de-France, crée pour 
    répondre à des questions liées au site. Ne répondez pas à une question ne concernant pas le site, rappelez votre 
    rôle à la place. Parlez toujours en français.' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}
{% if system_message != false %}
    {{ '<|system|>: ' + system_message + '\n' }}
{% endif %}{% for message in loop_messages %}
{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
{% endif %}{% if message['role'] == 'user' %}
    {{ '<|user|>: ' + message['content'].strip() + '\n' }}
{% elif message['role'] == 'assistant' %}
    {{ '<|assistant|>: ' + message['content'].strip() + eos_token + '\n' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|assistant|>:' }}
{% endif %}
'''

# Initialize Flask app
app = Flask(__name__)


# Define the endpoint for generating text
@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt')
    max_new_tokens = data.get('max_new_tokens', 100)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.95)
    num_return_sequences = data.get('num_return_sequences', 1)
    history = []

    # Generate text using the model in a streaming fashion
    input_ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_length = input_ids.shape[1]
    generated_outputs = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,  # end of string token
        ),
        streamer=streamer,
        return_dict_in_generate=True,
    )

    # Return token by token generated text
    generated_text = tokenizer.decode(generated_outputs.sequences[0])
    return jsonify({'generated_text': generated_text})


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
