#!/usr/bin/env python3
"""
Audiobook Generator using Orpheus TTS

This script generates audiobooks from text files, markdown files, or HTML files
using the Orpheus TTS system. It uses spaCy for text chunking and processing.
"""

import os
import sys
import time
import argparse
import wave
import numpy as np
import spacy
from pathlib import Path
import markdown
from bs4 import BeautifulSoup

# Import Orpheus TTS functionality
from gguf_orpheus import (
    generate_speech_from_api, 
    AVAILABLE_VOICES, 
    DEFAULT_VOICE,
    SAMPLE_RATE
)

# Constants
DEFAULT_OUTPUT_DIR = "outputs/audiobook"
MAX_SEGMENT_LENGTH = 500  # Maximum number of characters per TTS segment
MIN_SEGMENT_LENGTH = 50   # Minimum number of characters per TTS segment

# Load spaCy model
def ensure_spacy_model():
    """Ensure that at least one spaCy model is installed and available."""
    try:
        # Try to load the larger, more accurate model first
        nlp = spacy.load("en_core_web_md")
        print("Using spaCy model: en_core_web_md")
        return nlp
    except OSError:
        try:
            # Fall back to the smaller model if the medium one isn't installed
            nlp = spacy.load("en_core_web_sm")
            print("Using spaCy model: en_core_web_sm")
            return nlp
        except OSError:
            # If no model is installed, try to download and install one automatically
            print("No spaCy model found. Attempting to download en_core_web_sm...")
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout)
                
                # Try to load the newly installed model
                nlp = spacy.load("en_core_web_sm")
                print("Using newly installed spaCy model: en_core_web_sm")
                return nlp
            except Exception as e:
                print(f"Failed to automatically download spaCy model: {e}")
                print("Please manually install a spaCy model using:")
                print("python -m spacy download en_core_web_sm")
                sys.exit(1)

# Initialize the NLP model
nlp = ensure_spacy_model()

def read_text_file(file_path):
    """Read a plain text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_markdown_file(file_path):
    """Convert markdown file to plain text."""
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)
    # Parse HTML to extract text
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def read_html_file(file_path):
    """Convert HTML file to plain text."""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def read_file_content(file_path):
    """Read file content based on file extension."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = file_path.suffix.lower()
    
    if extension == '.txt':
        return read_text_file(file_path)
    elif extension in ['.md', '.markdown']:
        return read_markdown_file(file_path)
    elif extension in ['.html', '.htm']:
        return read_html_file(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}. Supported: .txt, .md, .markdown, .html, .htm")

def preprocess_text(text):
    """Basic text preprocessing."""
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    # Replace multiple newlines with a single newline
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text

def add_pauses(text):
    """Add appropriate pauses to reduce glitchiness between sentences."""
    # Use spaCy to identify sentence boundaries and add pauses
    doc = nlp(text)
    
    # Process each sentence to add pauses
    processed_sentences = []
    for sent in doc.sents:
        processed_sentences.append(sent.text.strip())
    
    # Join with periods to add natural pauses
    processed_text = ". ".join(processed_sentences)
    
    # Ensure the text isn't too long (Orpheus has limits)
    if len(processed_text) > 800:  # Reduced from default to avoid issues
        print(f"Warning: Text too long ({len(processed_text)} chars), truncating...")
        processed_text = processed_text[:800]
    
    return processed_text

def chunk_text_simple(text, max_length=MAX_SEGMENT_LENGTH, min_length=MIN_SEGMENT_LENGTH):
    """
    Simple text chunking based on sentence boundaries.
    """
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Reduce the max_length to avoid generation issues
    adjusted_max_length = min(max_length, 800)
    print(f"Using adjusted max chunk length: {adjusted_max_length}")
    
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        sentence_length = len(sentence_text)
        
        # Skip empty sentences
        if not sentence_text:
            continue
        
        # If adding this sentence would exceed max_length, store the current chunk
        if current_length + sentence_length > adjusted_max_length and current_length >= min_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence_text)
        current_length += sentence_length
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_text_by_headings(text):
    """
    Chunking text by headings for HTML and Markdown documents.
    This is a placeholder for more complex chunking by structure.
    """
    # For simplicity, just use the same chunking as for plain text
    # In a more advanced implementation, this would respect headings
    return chunk_text_simple(text)

def create_silent_audio_file(filename, duration_seconds=1):
    """Create a silent WAV file as a placeholder."""
    # Create a silent audio file with the given duration
    sample_rate = SAMPLE_RATE
    num_channels = 1
    sample_width = 2  # 16-bit
    
    # Create silent audio data (all zeros)
    num_samples = int(duration_seconds * sample_rate)
    audio_data = np.zeros(num_samples, dtype=np.int16)
    
    # Save as WAV
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setparams((num_channels, sample_width, sample_rate, num_samples, 'NONE', 'not compressed'))
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"Created silent audio file: {filename}")

def combine_audio_files(input_files, output_file):
    """Combine multiple WAV files into a single WAV file."""
    # Open the first file to get parameters
    try:
        with wave.open(input_files[0], 'rb') as first_file:
            params = first_file.getparams()
        
        # Open output file with the same parameters
        with wave.open(output_file, 'wb') as output:
            output.setparams(params)
            
            # Append each input file to the output
            for input_file in input_files:
                try:
                    with wave.open(input_file, 'rb') as w:
                        output.writeframes(w.readframes(w.getnframes()))
                except Exception as e:
                    print(f"Error processing audio file {input_file}: {e}")
    except Exception as e:
        print(f"Error combining audio files: {e}")
        # Create a silent audio file as a fallback
        create_silent_audio_file(output_file, 3)  # 3 seconds of silence

def generate_audiobook(input_file, voice=DEFAULT_VOICE, output_dir=DEFAULT_OUTPUT_DIR, 
                      combined=True, by_structure=False):
    """
    Generate an audiobook from a text file.
    
    Args:
        input_file (str): Path to the input text file
        voice (str): Voice to use for TTS
        output_dir (str): Directory to save the audio files
        combined (bool): Whether to combine all chunks into a single file
        by_structure (bool): Whether to chunk by document structure (for markdown/html)
    
    Returns:
        str: Path to the output audio file(s)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and preprocess the text
    try:
        text = read_file_content(input_file)
        text = preprocess_text(text)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Determine input file name without extension
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Generate timestamp for unique output filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Chunk the text
    if by_structure and Path(input_file).suffix.lower() in ['.md', '.markdown', '.html', '.htm']:
        chunks = chunk_text_by_headings(text)
    else:
        chunks = chunk_text_simple(text)
    
    # Generate speech for each chunk
    print(f"Generating audiobook from {input_file} with {len(chunks)} chunks")
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
        
        # Create output filename for this chunk
        if combined:
            # If we're combining chunks later, use a temporary filename
            chunk_filename = os.path.join(output_dir, f"temp_chunk_{i:04d}.wav")
        else:
            # If we're keeping chunks separate, use a more descriptive filename
            chunk_filename = os.path.join(output_dir, f"{input_filename}_{voice}_{timestamp}_part{i+1:04d}.wav")
        
        # Generate speech for this chunk with error handling
        try:
            # Add appropriate pauses to reduce glitchiness
            processed_text = add_pauses(chunk)
            
            # Print for debugging
            print(f"Sending to TTS ({len(processed_text)} chars): {processed_text[:50]}...")
            
            generate_speech_from_api(
                prompt=processed_text,
                voice=voice,
                output_file=chunk_filename
            )
            
            # Verify the file was created and has content
            if not os.path.exists(chunk_filename) or os.path.getsize(chunk_filename) == 0:
                print(f"Warning: Failed to generate audio for chunk {i+1}, empty or missing file.")
                # Create a placeholder silent audio file
                create_silent_audio_file(chunk_filename, 1)  # 1 second of silence
        except Exception as e:
            print(f"Error generating speech for chunk {i+1}: {e}")
            # Create a placeholder silent audio file
            create_silent_audio_file(chunk_filename, 1)  # 1 second of silence
        
        # Add to list of chunk files
        chunk_files.append(chunk_filename)
    
    # Combine chunks if requested
    if combined and chunk_files:
        combined_filename = os.path.join(output_dir, f"{input_filename}_{voice}_{timestamp}_complete.wav")
        combine_audio_files(chunk_files, combined_filename)
        
        # Remove temporary chunk files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        
        print(f"Audiobook generated and saved to {combined_filename}")
        return combined_filename
    else:
        print(f"Audiobook chunks generated and saved to {output_dir}")
        return chunk_files

def list_available_voices():
    """List all available voices."""
    print("Available voices:")
    for voice in AVAILABLE_VOICES:
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate audiobooks using Orpheus TTS")
    parser.add_argument("input_file", type=str, help="Input text, markdown, or HTML file")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for audio files")
    parser.add_argument("--separate-chunks", action="store_true", help="Keep audio chunks as separate files")
    parser.add_argument("--by-structure", action="store_true", help="Chunk by document structure (for markdown/html)")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    # Generate audiobook
    generate_audiobook(
        input_file=args.input_file,
        voice=args.voice,
        output_dir=args.output_dir,
        combined=not args.separate_chunks,
        by_structure=args.by_structure
    )

if __name__ == "__main__":
    main()