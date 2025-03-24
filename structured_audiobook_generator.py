#!/usr/bin/env python3
"""
Structured Audiobook Generator using Orpheus TTS

An enhanced audiobook generator that provides sophisticated chunking mechanisms
for structured documents (Markdown, HTML) and plain text files.
It uses spaCy for NLP-based chunking and respects document structure.
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
import re
import json

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

class AudiobookChunk:
    """
    Represents a chunk of text in the audiobook with metadata.
    """
    def __init__(self, text, level=0, title=None, order=0):
        self.text = text
        self.level = level  # Heading level (0 for body text)
        self.title = title  # Section title
        self.order = order  # Position in document
        self.audio_file = None  # Path to generated audio file
        
    def __str__(self):
        prefix = f"{'#' * self.level} {self.title}: " if self.title else ""
        return f"{prefix}{self.text[:50]}..." if len(self.text) > 50 else self.text

class AudiobookProcessor:
    """
    Processes text files into audiobook chunks and generates speech.
    """
    def __init__(self, voice=DEFAULT_VOICE, output_dir=DEFAULT_OUTPUT_DIR, 
                max_length=MAX_SEGMENT_LENGTH, min_length=MIN_SEGMENT_LENGTH):
        self.voice = voice
        self.output_dir = output_dir
        self.max_length = max_length
        self.min_length = min_length
        self.chunks = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def read_file(self, file_path):
        """Read file based on file extension and process accordingly."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        print(f"Processing file with extension: {extension}")
        
        if extension == '.txt':
            self.chunks = self._process_text_file(file_path)
        elif extension in ['.md', '.markdown']:
            self.chunks = self._process_markdown_file(file_path)
        elif extension in ['.html', '.htm']:
            self.chunks = self._process_html_file(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}. Supported: .txt, .md, .markdown, .html, .htm")
        
        print(f"Created {len(self.chunks)} chunks for processing")
        return self.chunks
    
    def _process_text_file(self, file_path):
        """Process a plain text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Preprocess text and create chunks
        text = self._preprocess_text(text)
        self.chunks = self._create_nlp_chunks(text)
        return self.chunks
    
    def _process_markdown_file(self, file_path):
        """Process a markdown file with structure preservation."""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
        
        print(f"Read markdown file with {len(md_content)} characters")
        
        # Extract structure from markdown
        chunks = self._process_markdown_content(md_content)
        print(f"Processed markdown content into {len(chunks)} chunks")
        return chunks
    
    def _process_html_file(self, file_path):
        """Process an HTML file with structure preservation."""
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        self.chunks = self._process_html_structure(soup)
        return self.chunks
    
    def _preprocess_text(self, text):
        """Clean and normalize text."""
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        # Replace multiple newlines with a single newline
        text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
        return text
    
    def _create_nlp_chunks(self, text):
        """Create chunks using NLP for optimal TTS segmentation."""
        doc = nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_order = 0
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            sentence_length = len(sentence_text)
            
            # Skip empty sentences
            if not sentence_text:
                continue
            
            # If adding this sentence would exceed max_length and we have enough text,
            # store the current chunk and start a new one
            if current_length + sentence_length > self.max_length and current_length >= self.min_length:
                chunks.append(AudiobookChunk(' '.join(current_chunk), order=chunk_order))
                chunk_order += 1
                current_chunk = []
                current_length = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence_text)
            current_length += sentence_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(AudiobookChunk(' '.join(current_chunk), order=chunk_order))
        
        return chunks
    
    def _process_markdown_content(self, md_content):
        """Process markdown content preserving structure."""
        # Convert markdown to HTML for easier processing
        html_content = markdown.markdown(md_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Process the HTML structure
        return self._process_html_structure(soup)
    
    def _process_html_structure(self, soup):
        """Process HTML structure respecting headings and paragraphs."""
        chunks = []
        current_heading = None
        current_level = 0
        current_text_chunks = []
        chunk_order = 0
        
        # Find all heading and paragraph elements
        elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li'])
        
        for element in elements:
            tag_name = element.name
            
            # Process headings
            if tag_name.startswith('h') and len(tag_name) == 2:
                # If we have accumulated text chunks under the previous heading,
                # process them now
                if current_text_chunks:
                    text_chunks = self._create_nlp_chunks(' '.join(current_text_chunks))
                    for chunk in text_chunks:
                        chunk.title = current_heading
                        chunk.level = current_level
                        chunk.order = chunk_order
                        chunk_order += 1
                        chunks.append(chunk)
                    current_text_chunks = []
                
                # Set the new current heading
                current_heading = element.get_text().strip()
                current_level = int(tag_name[1])
                
                # Add the heading itself as a chunk
                chunks.append(AudiobookChunk(
                    current_heading, 
                    level=current_level, 
                    title=current_heading,
                    order=chunk_order
                ))
                chunk_order += 1
            
            # Process paragraphs and list items
            elif tag_name in ['p', 'li']:
                text = element.get_text().strip()
                if text:
                    current_text_chunks.append(text)
        
        # Process any remaining text chunks
        if current_text_chunks:
            text_chunks = self._create_nlp_chunks(' '.join(current_text_chunks))
            for chunk in text_chunks:
                chunk.title = current_heading
                chunk.level = current_level
                chunk.order = chunk_order
                chunk_order += 1
                chunks.append(chunk)
        
        return chunks
    
    def generate_audio(self, input_filename, combined=True):
        """Generate audio for all chunks and optionally combine them."""
        if not self.chunks:
            raise ValueError("No chunks available. Process a file first.")
        
        # Print detailed information about chunks for debugging
        print(f"Chunks information:")
        for i, chunk in enumerate(self.chunks):
            print(f"  Chunk {i+1}: {len(chunk.text)} chars, title: '{chunk.title}', level: {chunk.level}")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        chunk_files = []
        
        print(f"Generating audio for {len(self.chunks)} chunks...")
        
        # Generate speech for each chunk
        for i, chunk in enumerate(self.chunks):
            print(f"Processing chunk {i+1}/{len(self.chunks)}: {chunk}")
            
            # Create output filename for this chunk
            if combined:
                chunk_filename = os.path.join(self.output_dir, f"temp_chunk_{i:04d}.wav")
            else:
                # Create a more descriptive filename for separate chunks
                title_part = f"_{re.sub(r'[^\w]', '_', chunk.title[:20] if chunk.title else '')}"
                chunk_filename = os.path.join(
                    self.output_dir, 
                    f"{input_filename}_{self.voice}{title_part}_{timestamp}_part{i+1:04d}.wav"
                )
            
            # Generate speech with error handling
            try:
                # Add appropriate pauses to reduce glitchiness
                processed_text = self._add_pauses(chunk.text)
                
                # Print the text being sent to TTS for debugging
                print(f"Sending to TTS ({len(processed_text)} chars): {processed_text[:50]}...")
                
                generate_speech_from_api(
                    prompt=processed_text,
                    voice=self.voice,
                    output_file=chunk_filename
                )
                
                # Verify the file was created and has content
                if not os.path.exists(chunk_filename) or os.path.getsize(chunk_filename) == 0:
                    print(f"Warning: Failed to generate audio for chunk {i+1}, empty or missing file.")
                    # Create a placeholder silent audio file if needed
                    self._create_silent_audio_file(chunk_filename, 1)  # 1 second of silence
            except Exception as e:
                print(f"Error generating speech for chunk {i+1}: {e}")
                # Create a placeholder silent audio file
                self._create_silent_audio_file(chunk_filename, 1)  # 1 second of silence
            
            # Store the audio file path
            chunk.audio_file = chunk_filename
            chunk_files.append(chunk_filename)
        
        # Combine chunks if requested
        if combined and chunk_files:
            combined_filename = os.path.join(
                self.output_dir, 
                f"{input_filename}_{self.voice}_{timestamp}_complete.wav"
            )
            self._combine_audio_files(chunk_files, combined_filename)
            
            # Remove temporary chunk files
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            
            # Create a metadata file with chunk information
            metadata_filename = os.path.join(
                self.output_dir, 
                f"{input_filename}_{self.voice}_{timestamp}_metadata.json"
            )
            self._save_metadata(metadata_filename, combined_filename)
            
            print(f"Audiobook generated and saved to {combined_filename}")
            print(f"Metadata saved to {metadata_filename}")
            return combined_filename
        else:
            # Create a metadata file with chunk information
            metadata_filename = os.path.join(
                self.output_dir, 
                f"{input_filename}_{self.voice}_{timestamp}_metadata.json"
            )
            self._save_metadata(metadata_filename)
            
            print(f"Audiobook chunks generated and saved to {self.output_dir}")
            print(f"Metadata saved to {metadata_filename}")
            return chunk_files
    
    def _combine_audio_files(self, input_files, output_file):
        """Combine multiple WAV files into a single WAV file."""
        # Open the first file to get parameters
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
                    print(f"Error processing file {input_file}: {e}")
    
    def _add_pauses(self, text):
        """Add appropriate pauses to reduce glitchiness between sentences."""
        # Use spaCy to identify sentence boundaries and add pauses
        doc = nlp(text)
        
        # Process each sentence to add pauses
        processed_sentences = []
        for sent in doc.sents:
            processed_sentences.append(sent.text.strip())
        
        # Join with commas and periods to add natural pauses
        processed_text = ". ".join(processed_sentences)
        
        # Ensure the text isn't too long (Orpheus has limits)
        if len(processed_text) > 800:  # Reduced from default to avoid issues
            # If too long, split into shorter chunks and take the first chunk
            # (we're already working with chunks, so this is a safety measure)
            print(f"Warning: Text too long ({len(processed_text)} chars), truncating...")
            processed_text = processed_text[:800]
        
        return processed_text
    
    def _create_silent_audio_file(self, filename, duration_seconds=1):
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
    
    def _save_metadata(self, metadata_filename, combined_file=None):
        """Save metadata about the audiobook chunks."""
        metadata = {
            "voice": self.voice,
            "combined_file": combined_file,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": []
        }
        
        for chunk in self.chunks:
            metadata["chunks"].append({
                "text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                "level": chunk.level,
                "title": chunk.title,
                "order": chunk.order,
                "audio_file": os.path.basename(chunk.audio_file) if chunk.audio_file else None
            })
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

def list_available_voices():
    """List all available voices."""
    print("Available voices:")
    for voice in AVAILABLE_VOICES:
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate structured audiobooks using Orpheus TTS")
    parser.add_argument("input_file", type=str, help="Input text, markdown, or HTML file")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for audio files")
    parser.add_argument("--separate-chunks", action="store_true", help="Keep audio chunks as separate files")
    parser.add_argument("--max-length", type=int, default=MAX_SEGMENT_LENGTH, help="Maximum characters per chunk")
    parser.add_argument("--min-length", type=int, default=MIN_SEGMENT_LENGTH, help="Minimum characters per chunk")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    # Create processor and generate audiobook
    processor = AudiobookProcessor(
        voice=args.voice,
        output_dir=args.output_dir,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Get input filename without extension
    input_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # Process file and generate audio
    try:
        processor.read_file(args.input_file)
        processor.generate_audio(input_filename, combined=not args.separate_chunks)
    except Exception as e:
        print(f"Error generating audiobook: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())