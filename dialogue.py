import argparse
from datetime import datetime
from math import ceil, log10
import os
from pathlib import Path
import sys
import time
from gguf_orpheus import generate_speech_from_api, TEMPERATURE, TOP_P, REPETITION_PENALTY, OUTPUT_DIR

import wave
import audioop
import numpy as np


def merge_wave_files(wave_files, output_file):
    # Open the first file to get parameters
    with wave.open(wave_files[0], 'rb') as wf:
        params = wf.getparams()
        nchannels, sampwidth, framerate = params[:3]

    # Create output file with same parameters
    output = wave.open(output_file, 'wb')
    output.setparams((nchannels, sampwidth, framerate,
                     0, 'NONE', 'not compressed'))

    # Merge the files
    try:
        for wave_file in wave_files:
            with wave.open(wave_file, 'rb') as wf:
                # Read and write all frames from this file
                frames = wf.readframes(wf.getnframes())
                output.writeframes(frames)
    finally:
        output.close()


def generate_dialogue(prompt, temperature, top_p, repetition_penalty, output_file):

    prompt = prompt.split('\n')
    dialogue = []
    for line in prompt:
        line = line.strip()
        if line:
            speaker, speech = line.split(':')
            dialogue.append((speaker.strip(), speech.strip()))

    wave_files = []
    tmpdir = f'{OUTPUT_DIR}/{Path(output_file).stem}_tmp'
    os.makedirs(tmpdir, exist_ok=True)
    for i, (speaker, speech) in enumerate(dialogue):

        padding_width = ceil(log10(len(dialogue)))

        output_tmpfile = f'{tmpdir}/{i:0{padding_width}}.wav'
        if not os.path.exists(output_tmpfile):
            generate_speech_from_api(
                speech,
                voice=speaker,
                output_file=output_tmpfile,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        wave_files.append(output_tmpfile)

    # merge wave files
    merge_wave_files(wave_files, output_file)
    # remove tmp dir
    for f in wave_files:
        os.remove(f)
    os.rmdir(tmpdir)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Orpheus Text-to-Speech Dialogue using LM Studio API")
    parser.add_argument("--input-text", type=str,
                        help="Text to convert to speech")
    parser.add_argument("--input-file", type=str,
                        help="Text file to convert to speech")
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument("--temperature", type=float,
                        default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P,
                        help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY,
                        help="Repetition penalty (>=1.1 required for stable generation)")

    args = parser.parse_args()

    # Use text from command line or prompt user
    input_text = args.input_text
    input_file = args.input_file
    if not input_text and not input_file:
        print("Please provide text to convert to speech using --input-text or --input-file")
        sys.exit(1)

    if not input_text and input_file:
        with open(input_file, 'r') as f:
            input_text = f.read()

    # Default output file if none provided
    output_file = args.output
    if not output_file:
        # Create outputs directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Generate a filename based on the voice and a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"{OUTPUT_DIR}/dialogue_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")

    # Generate speech
    start_time = time.time()
    generate_dialogue(
        prompt=input_text,
        output_file=output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    end_time = time.time()

    print(
        f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")


if __name__ == '__main__':
    main()
