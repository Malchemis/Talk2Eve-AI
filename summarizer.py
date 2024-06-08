from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# Initialize BART model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


def chunk_text(text, sliding_window=128, overlap=32):
    """
    Chunks the given text into smaller pieces.

    Args:
    text (str): The text to be chunked.
    sliding_window (int): The maximum length of each chunk. Defaults to 1024 to fit within BART's input size.

    Returns:
    List[str]: The list of text chunks.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + sliding_window, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        start += sliding_window - overlap
    return chunks


def _summarize(text, max_length=32, min_length=16):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']


def summarize(text, max_length=512, min_length=64):
    """
    Summarizes the given text using the BART large CNN model.

    Args:
    text (str): The text to be summarized.
    max_length (int): The maximum length of the summary.
    min_length (int): The minimum length of the summary.

    Returns:
    str: The summarized text.
    """
    chunks = chunk_text(text)
    summaries = [_summarize(chunk) for chunk in chunks]
    summary = ' '.join(summaries)
    # post process the summary by re-feeding it to the summarizer
    return _summarize(summary, max_length, min_length)


# Example usage
if __name__ == "__main__":
    example_text = """
        The MITRE ATT&CK framework is a comprehensive matrix of tactics and techniques used by cyber adversaries.
        It is used by organizations to better understand the behavior of cyber attackers and to improve their security 
        posture.
        The framework is divided into several categories including Initial Access, Execution, Persistence, Privilege 
        Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Exfiltration, and 
        Impact. Each category contains specific techniques that adversaries use to achieve their objectives. 
        For example, under the Execution category, techniques include Command and Scripting Interpreter, PowerShell, 
        and Exploitation for Client Execution. Security teams can use the framework to identify gaps in their defenses 
        and to develop strategies to detect, respond to, and mitigate cyber threats.
    """
    print(summarize(example_text))
