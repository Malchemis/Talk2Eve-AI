from transformers import pipeline
import os

os.environ["TRANSFORMERS_CACHE"] = "/media/malchemis/CYBERTRON/models"


def summarize(text, max_length=130, min_length=30,
              model="facebook/bart-large-cnn", cache_dir="/media/malchemis/CYBERTRON/models"):
    """
    Summarizes the given text using the BART large CNN model.

    Args:
    text (str): The text to be summarized.
    max_length (int): The maximum length of the summary.
    min_length (int): The minimum length of the summary.

    Returns:
    str: The summarized text.
    """
    # Load summarization pipeline
    summarizer = pipeline("summarization", model=model)

    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


# Example usage
if __name__ == "__main__":
    example_text = """
        The MITRE ATT&CK framework is a comprehensive matrix of tactics and techniques used by cyber adversaries.
        It is used by organizations to better understand the behavior of cyber attackers and to improve their security posture.
        The framework is divided into several categories including Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Exfiltration, and Impact.
        Each category contains specific techniques that adversaries use to achieve their objectives. For example, under the Execution category, techniques include Command and Scripting Interpreter, PowerShell, and Exploitation for Client Execution.
        Security teams can use the framework to identify gaps in their defenses and to develop strategies to detect, respond to, and mitigate cyber threats.
    """
    print(summarize(example_text))
