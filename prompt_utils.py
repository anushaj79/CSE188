import json
import random
import re
import requests
from openai import OpenAI
import anthropic
import google.generativeai as genai
from groq import Groq
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY

# API CLIENT INITIALIZATION

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


# DATASET UTILITIES

def load_dataset(url):
    """Load dataset from URL and return as list."""
    response = requests.get(url)
    response.raise_for_status()
    data = json.loads(response.text)
    return list(data.values())

# MODEL PROMPTING

def prompt_gpt(question, instruction):
    """Send prompt to GPT-4 and return response."""
    completion = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return completion.choices[0].message.content.strip()

def prompt_claude(question, instruction):
    """Send prompt to Claude and return response."""
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        system=instruction,
        messages=[{"role": "user", "content": question}],
        max_tokens=1024
    )
    return response.content[0].text.strip()

def prompt_gemini(question, instruction):
    """Send prompt to Gemini 1.5 Flash and return response."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    full_prompt = f"{instruction}\n\n{question}"
    
    response = model.generate_content(full_prompt)
    return response.text.strip()

def prompt_groq(question, instruction):
    """Send prompt to Llama 3 via Groq and return response."""
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return completion.choices[0].message.content.strip()

# TEXT PROCESSING AND EXTRACTION

def extract_answer_tag(text):
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def split_steps(full_output):
    """Extract steps from [Steps] section as a list of strings."""
    steps_match = re.search(r"\[Steps\](.*)\[Answer\]", full_output, re.DOTALL)
    if steps_match:
        steps_text = steps_match.group(1).strip()
        # Split on <step>...</step> tags
        step_lines = re.findall(r"<step>(.*?)</step>", steps_text, re.DOTALL)
        return [s.strip() for s in step_lines]
    return []