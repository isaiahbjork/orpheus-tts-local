import os
import sys
import requests
import json
import time
import wave
import numpy as np
import sounddevice as sd
import argparse
import threading
import queue
import asyncio

# LM Studio API settings
API_URL = "http://127.0.0.1:1234/v1/completions"
HEADERS = {
    "Content-Type": "application/json"
}

# Model parameters
MAX_TOKENS = 1200
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"  # Best voice according to documentation
OUTPUT_DIR = "outputs"

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX = "<custom_token_"

def format_prompt(prompt, voice=DEFAULT_VOICE):
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
        
    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"
    
    # Add special token markers for the LM Studio API
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"   # Using the eos_token from config
    
    return f"{special_start}{formatted_prompt}{special_end}"

def generate_tokens_from_api(prompt, voice=DEFAULT_VOICE, temperature=TEMPERATURE, 
                            top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    """Generate tokens from text using LM Studio API."""
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    # Create the request payload for the LM Studio API
    payload = {
        "model": "orpheus-3b-0.1-ft-q4_k_m",  # Model name can be anything, LM Studio ignores it
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }
    
    # Make the API request with streaming
    response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)
    
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Error details: {response.text}")
        return
    
    # Process the streamed response
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
                            yield token_text
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue
    
    print("Token generation complete")

def turn_token_into_id(token_string, index):
    """Convert token string to numeric ID for audio processing."""
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None

def convert_to_audio(multiframe, count):
    """Convert token frames to audio."""
    # Import here to avoid circular imports
    from decoder import convert_to_audio as orpheus_convert_to_audio
    return orpheus_convert_to_audio(multiframe, count)

async def tokens_decoder(token_gen):
    """Asynchronous token decoder that converts token stream to audio stream."""
    buffer = []
    count = 0
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            
            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """Synchronous wrapper for the asynchronous token decoder."""
    audio_queue = queue.Queue()
    audio_segments = []
    
    # If output_file is provided, prepare WAV file
    wav_file = None
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    # Convert the synchronous token generator into an async generator
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel to indicate completion

    def run_async():
        asyncio.run(async_producer())

    # Start the async producer in a separate thread
    thread = threading.Thread(target=run_async)
    thread.start()

    # Process audio as it becomes available
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        audio_segments.append(audio)
        
        # Write to WAV file if provided
        if wav_file:
            wav_file.writeframes(audio)
    
    # Close WAV file if opened
    if wav_file:
        wav_file.close()
    
    thread.join()
    
    # Calculate and print duration
    duration = sum([len(segment) // (2 * 1) for segment in audio_segments]) / SAMPLE_RATE
    print(f"Generated {len(audio_segments)} audio segments")
    print(f"Generated {duration:.2f} seconds of audio")
    
    return audio_segments

def stream_audio(audio_buffer):
    """Stream audio buffer to output device."""
    if audio_buffer is None or len(audio_buffer) == 0:
        return
    
    # Convert bytes to NumPy array (16-bit PCM)
    audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
    
    # Normalize to float in range [-1, 1] for playback
    audio_float = audio_data.astype(np.float32) / 32767.0
    
    # Play the audio
    sd.play(audio_float, SAMPLE_RATE)
    sd.wait()

def generate_speech_from_api(prompt, voice=DEFAULT_VOICE, output_file=None, temperature=TEMPERATURE, 
                     top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    """Generate speech from text using Orpheus model via LM Studio API."""
    return tokens_decoder_sync(
        generate_tokens_from_api(
            prompt=prompt, 
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        ),
        output_file=output_file
    )

def list_available_voices():
    """List all available voices with the recommended one marked."""
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")
    
    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech using LM Studio API")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, 
                       help="Repetition penalty (>=1.1 required for stable generation)")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    # Use text from command line or prompt user
    prompt = args.text
    if not prompt:
        if len(sys.argv) > 1 and sys.argv[1] not in ("--voice", "--output", "--temperature", "--top_p", "--repetition_penalty"):
            prompt = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        else:
            prompt = input("Enter text to synthesize: ")
            if not prompt:
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."
    
    # Default output file if none provided
    output_file = args.output
    if not output_file:
        # Create outputs directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Generate a filename based on the voice and a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"{OUTPUT_DIR}/{args.voice}_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")
    
    # Generate speech
    start_time = time.time()
    audio_segments = generate_speech_from_api(
        prompt=prompt,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_file=output_file
    )
    end_time = time.time()
    
    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    main() 