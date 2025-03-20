import requests
import json

from common import format_prompt, DEFAULT_VOICE, MAX_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY, HEADERS


OLLAMA_API_URL = "http://localhost:11434/v1/completions"
TTS_MODEL = "isaiabjork/orpheus-tts:3b-Q4_K_M"

def generate_tokens_from_api(prompt, voice=DEFAULT_VOICE, temperature=TEMPERATURE, 
                            top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    # Construct payload based on Ollama's API requirements
    payload = {
        "model": TTS_MODEL,
        "prompt": formatted_prompt,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
        },
        "stream": True
    }

    response = requests.post(OLLAMA_API_URL, headers=HEADERS, json=payload, stream=True)
    
    token_counter = 0
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]  # Remove the 'data: ' prefix
                if data_str.strip() == '[DONE]':
                    break
                    
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        token_text = data['choices'][0].get('text', '')
                        token_counter += 1
                        if token_text:
                            # print("text: ", token_text)
                            yield token_text
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

        
    print("Token generation complete")
