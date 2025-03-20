import requests

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
        "prompt": prompt,
        "voice": voice,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        
        print(response.text)

        # Parse the JSON response from Ollama
        token_data = response.json()
        
        for token in token_data.get('tokens', []):
            yield token
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return

