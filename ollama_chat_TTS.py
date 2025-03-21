#!/usr/bin/env python3

import ollama
import sounddevice as sd
import time
import numpy as np
import argparse
from gguf_orpheus import generate_speech_from_api

def get_llm_response(prompt, model):
    system_prompt = """You are a creative text generator that MUST include emotion tags in your responses.
    You MUST use at least 3 emotion tags in every response, properly formatted with <> brackets.
    Always insert these emotion tags naturally within your text to convey emotions and sounds.
    Use only these approved tags:

<giggle>
<laugh>
<chuckle>
<sigh>
<cough>
<sniffle>
<groan>
<yawn>
<gasp>

Never use any other format like **, (), or [] for these emotions.
Place the tags where they naturally fit in the conversation flow.
Example: "Hello! <giggle> I'm so excited to talk to you! <laugh> This is going to be fun!"
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt}, 
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return None

def play_audio(audio_segments):
    """Play audio segments using sounddevice."""
    if not audio_segments:
        return
    
    # Convert audio segments to numpy array and play
    try:
        # Directly concatenate the bytes data from audio segments
        audio_data = np.frombuffer(b''.join(audio_segments), dtype=np.int16)
        # Normalize audio to prevent clipping
        audio_data = audio_data.astype(np.float32) / 32768.0
        # Add a slight delay between words for more natural pacing
        silence = np.zeros(int(24000 * 0.1))  # 0.1 second silence
        audio_data = np.concatenate([audio_data, silence])
        # Play audio with adjusted settings
        sd.play(audio_data, samplerate=24000, blocking=True)
        sd.wait()  # Wait until audio is finished playing
    except Exception as e:
        print(f"Error playing audio: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time Text-to-Speech System with Ollama and Orpheus")
    parser.add_argument("--model", type=str, default="llama2:3b", help="Ollama model to use (default: llama2:3b)")
    parser.add_argument("--voice", type=str, default="tara", help="Voice to use for TTS (default: tara)")
    args = parser.parse_args()

    print("Real-time Text-to-Speech System")
    print(f"Using model: {args.model}")
    print(f"Using voice: {args.voice}")
    print("Enter your text (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                continue
            
            # Get response from LLM
            print("Generating response...")
            llm_response = get_llm_response(user_input, args.model)
            
            if llm_response:
                print(f"\nAI: {llm_response}\n")
                
                # Convert response to speech
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                print("Converting to speech...")
                audio_segments = generate_speech_from_api(
                    prompt=llm_response,
                    voice=args.voice,
                    output_file=f"outputs/{args.voice}_{timestamp}.wav",
                    temperature=0.6,  # Lower temperature for more consistent output
                    top_p=0.9,  # Slightly higher top_p for more natural speech
                    repetition_penalty=1.1  # Increased repetition penalty for better clarity
                )
                
                # Play the generated audio
                print("Playing audio...")
                play_audio(audio_segments)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()