import requests
import json

from common import format_prompt, DEFAULT_VOICE, MAX_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY, HEADERS


OLLAMA_API_URL = "http://localhost:11434/api/generate"
TTS_MODEL = "isaiabjork/orpheus-tts:3b-Q4_K_M"

def generate_tokens_from_api(prompt, voice=DEFAULT_VOICE, temperature=TEMPERATURE, 
                            top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    # Construct payload based on Ollama's API requirements
    data = {
        "model": TTS_MODEL,
        "prompt": formatted_prompt,
        "num_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, headers=HEADERS, json=data, stream=True)
        response.raise_for_status()

        # Process streamed JSON responses
        for line in response.iter_lines():
            print(line)
            if line:
                try:
                    token_data = json.loads(line)
                    if "response" in token_data:
                        yield token_data["response"]
                    if token_data.get("done"):
                        break  
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return