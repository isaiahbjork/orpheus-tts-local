#!/usr/bin/env python3
"""
Example script demonstrating how to use the audiobook generators.
This shows how to use both the simple and structured audiobook generators.
"""

import os
import time
from pathlib import Path

# Import the audiobook generators
from audiobook_generator import generate_audiobook
from structured_audiobook_generator import AudiobookProcessor

def create_sample_text_file():
    """Create a sample text file for testing."""
    sample_dir = Path("examples")
    sample_dir.mkdir(exist_ok=True)
    
    sample_file = sample_dir / "sample_text.txt"
    
    text = """
    The Orpheus Text-to-Speech System
    
    Artificial Intelligence has revolutionized many fields, and text-to-speech is no exception. 
    Modern TTS systems can generate highly realistic speech that's almost indistinguishable from human voices.
    
    The Orpheus TTS system represents a significant advancement in this area. It uses neural networks to generate
    natural, expressive speech with proper intonation and emotional nuances. This makes it ideal for creating
    audiobooks, voice assistants, and accessibility solutions.
    
    How Orpheus Works
    
    At its core, Orpheus uses a neural text-to-speech model that converts text into audio tokens.
    These tokens are then processed to generate realistic speech waveforms.
    
    The system supports multiple voices and can handle various emotions and speaking styles.
    This flexibility makes it suitable for a wide range of applications.
    
    Applications and Future Developments
    
    Audiobooks are just one application of this technology. Voice assistants, accessibility tools,
    and educational content can all benefit from high-quality TTS systems like Orpheus.
    
    As AI continues to evolve, we can expect even more natural-sounding TTS systems in the future.
    These advancements will further blur the line between synthesized and human speech.
    """
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return sample_file

def create_sample_markdown_file():
    """Create a sample markdown file for testing."""
    sample_dir = Path("examples")
    sample_dir.mkdir(exist_ok=True)
    
    sample_file = sample_dir / "sample_markdown.md"
    
    markdown_text = """
# The Orpheus Text-to-Speech System

Artificial Intelligence has revolutionized many fields, and text-to-speech is no exception. 
Modern TTS systems can generate highly realistic speech that's almost indistinguishable from human voices.

## Introduction to Orpheus

The Orpheus TTS system represents a significant advancement in this area. It uses neural networks to generate
natural, expressive speech with proper intonation and emotional nuances. This makes it ideal for creating
audiobooks, voice assistants, and accessibility solutions.

## How Orpheus Works

At its core, Orpheus uses a neural text-to-speech model that converts text into audio tokens.
These tokens are then processed to generate realistic speech waveforms.

The system supports multiple voices and can handle various emotions and speaking styles.
This flexibility makes it suitable for a wide range of applications.

## Applications and Future Developments

Audiobooks are just one application of this technology. Voice assistants, accessibility tools,
and educational content can all benefit from high-quality TTS systems like Orpheus.

As AI continues to evolve, we can expect even more natural-sounding TTS systems in the future.
These advancements will further blur the line between synthesized and human speech.

### Specific Applications

- **Audiobooks**: Create engaging audio content from written material
- **Voice Assistants**: Provide natural-sounding responses to user queries
- **Educational Content**: Make learning materials accessible to all students
- **Accessibility Tools**: Help people with visual impairments access written content
"""
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(markdown_text)
    
    return sample_file

def example_simple_audiobook():
    """Example of using the simple audiobook generator."""
    print("\n=== Simple Audiobook Generator Example ===\n")
    
    # Create a sample text file
    sample_file = create_sample_text_file()
    print(f"Created sample text file: {sample_file}")
    
    # Generate an audiobook from the text file
    output_dir = "examples/output_simple"
    start_time = time.time()
    
    # Override constants for the example to use smaller chunk sizes
    import audiobook_generator
    audiobook_generator.MAX_SEGMENT_LENGTH = 300  # Shorter chunks for the example
    audiobook_generator.MIN_SEGMENT_LENGTH = 50   # Minimum size
    
    output_file = generate_audiobook(
        input_file=sample_file,
        voice="tara",  # Using Tara voice
        output_dir=output_dir,
        combined=True  # Combine all chunks into a single file
    )
    
    end_time = time.time()
    print(f"Generated audiobook in {end_time - start_time:.2f} seconds")
    print(f"Output file: {output_file}")

def example_structured_audiobook():
    """Example of using the structured audiobook generator."""
    print("\n=== Structured Audiobook Generator Example ===\n")
    
    # Create a sample markdown file
    sample_file = create_sample_markdown_file()
    print(f"Created sample markdown file: {sample_file}")
    
    # Create output directory
    output_dir = "examples/output_structured"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the AudiobookProcessor
    processor = AudiobookProcessor(
        voice="leo",  # Using Leo voice
        output_dir=output_dir,
        max_length=200,  # Even shorter chunks for the example
        min_length=50
    )
    
    # Process the markdown file
    start_time = time.time()
    
    # Get input filename without extension
    input_filename = os.path.splitext(os.path.basename(sample_file))[0]
    
    # Process file and generate audio
    try:
        # This is the key fix - we need to properly store the returned chunks
        chunks = processor.read_file(sample_file)
        print(f"Processed markdown file and created {len(chunks)} chunks")
        
        output_file = processor.generate_audio(input_filename, combined=True)
        
        end_time = time.time()
        print(f"Generated structured audiobook in {end_time - start_time:.2f} seconds")
        print(f"Output file: {output_file}")
    except Exception as e:
        print(f"Error generating audiobook: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging

def main():
    print("Orpheus Audiobook Generator Examples")
    print("====================================")
    
    # Run examples
    example_simple_audiobook()
    example_structured_audiobook()
    
    print("\nExamples completed. Check the output directories for the generated audiobooks.")

if __name__ == "__main__":
    main()