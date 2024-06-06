"""
Dataset creation :
Uses Mistral 7b-instruct-v0.1-hf model to generate questions for all given files from a folder.
Parses the output so that troncated answers are only inclueded to the last full stop.
Saves to a csv file {input: question, output{document}}
Always add to the document the name and the link to the source. So that the user can refer to the original document.
"""

# Modules
import sys  # to read arguments
from pathlib import Path  # to handle files

# Load model and tokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

import pandas as pd


def main():
    # Read arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_questions.py input_folder output_file")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    model_name = '/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Generate questions
    questions_document = pd.DataFrame(columns=['input', 'output'])
    with open(output_file, 'w') as file:
        for input_file in files_generator(input_folder):
            with open(input_file, 'r') as f:
                document = f.read()
            gen_output = pipe(get_prompt(document),
                              do_sample=True,
                              max_new_tokens=100,
                              temperature=0.7,
                              top_k=50,
                              top_p=0.95,
                              num_return_sequences=1
            )[0]['generated_text']
            questions = gen_output.split('[Question(s)] :')[1].split('\n')[1:]


# Utils
def get_prompt(document):
    head = "\n[Tâche] :\nProduit 3 questions auxquelles le document répond. Sois précis.\n[Document] :\n"
    foot = "\n[Question(s)] :\n"
    return head + document + foot


def files_generator(input_folder: Path):
    for file in input_folder.iterdir():
        if file.is_file():
            yield file


def get_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


if __name__ == '__main__':
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
