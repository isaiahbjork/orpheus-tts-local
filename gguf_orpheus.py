import argparse
import asyncio
import json
import logging
import queue
import re
import sys
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Generator, Optional

import numpy as np
import requests
import sounddevice as sd

# Set up logging
logger = logging.getLogger(__name__)

# LM Studio API settings
API_URL = "http://127.0.0.1:1234/v1/completions"
HEADERS = {"Content-Type": "application/json"}

# Model parameters
MAX_TOKENS: int = 1200
TEMPERATURE: float = 0.6
TOP_P: float = 0.9
REPETITION_PENALTY: float = 1.1
SAMPLE_RATE: int = 24000  # SNAC model uses 24kHz
CHUNK_LIMIT: int = 400

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES: list[str] = [
    "tara",
    "leah",
    "jess",
    "leo",
    "dan",
    "mia",
    "zac",
    "zoe",
]
DEFAULT_VOICE: str = "tara"  # Best voice according to documentation

# Special token IDs for Orpheus model
START_TOKEN_ID: int = 128259
END_TOKEN_IDS: list[int] = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX: str = "<custom_token_"

# Available emotion tags
EMOTION_TAGS: set[str] = {
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
}


@dataclass
class TTSConfig:
    """Configuration for the TTS system."""

    voice: str = DEFAULT_VOICE
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_tokens: int = MAX_TOKENS
    repetition_penalty: float = REPETITION_PENALTY
    chunk_max_length: int = CHUNK_LIMIT


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Format prompt for Orpheus model with voice prefix and special tokens.

    Args:
        prompt: The text to convert to speech
        voice: The voice to use for synthesis

    Returns:
        A formatted prompt string ready for the model
    """
    if voice not in AVAILABLE_VOICES:
        logger.warning(
            f"Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead."
        )
        voice = DEFAULT_VOICE

    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"

    # Add special token markers for the LM Studio API
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"  # Using the eos_token from config

    return f"{special_start}{formatted_prompt}{special_end}"


def chunk_text(text: str, max_length: int = 400) -> list[str]:
    """
    Split text into chunks based on sentence delimiters (., !, ?).
    Each chunk will be at most max_length characters.

    Args:
        text: The input text to chunk
        max_length: Maximum length of each chunk

    Returns:
        List of text chunks
    """
    # Initialize variables
    chunks: list[str] = []
    current_chunk: str = ""

    # Split the text by sentence delimiters while keeping the delimiters
    sentences = re.findall(r"[^.!?]+[.!?](?:\s|$)", text + " ")

    for sentence in sentences:
        # If adding this sentence would exceed max_length, start a new chunk
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def generate_tokens_from_api(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
) -> Generator[str, None, None]:
    """
    Generate tokens from text using LM Studio API.

    Args:
        prompt: The text to convert to speech
        voice: The voice to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty

    Yields:
        Token strings from the API response
    """
    formatted_prompt = format_prompt(prompt, voice)
    logger.debug(f"Generating speech for: {formatted_prompt}")

    # Create the request payload for the LM Studio API
    payload: dict[str, Any] = {
        "model": "orpheus-3b-0.1-ft-q4_k_m",  # Model name can be anything, LM Studio ignores it
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True,
    }

    # Make the API request with streaming
    response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)

    if response.status_code != 200:
        logger.error(f"API request failed with status code {response.status_code}")
        logger.error(f"Error details: {response.text}")
        return

    # Process the streamed response
    token_counter = 0
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove the 'data: ' prefix
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    if "choices" in data and len(data["choices"]) > 0:
                        token_text = data["choices"][0].get("text", "")
                        token_counter += 1
                        if token_text:
                            yield token_text
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
                    continue

    logger.debug("Token generation complete")


def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """
    Convert token string to numeric ID for audio processing.

    Args:
        token_string: The token string to convert
        index: The current token index

    Returns:
        Token ID as integer, or None if conversion failed
    """
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


def convert_to_audio(multiframe: list[int], count: int) -> Optional[bytes]:
    """
    Convert token frames to audio.

    Args:
        multiframe: List of token IDs
        count: Current token count

    Returns:
        Audio data as bytes or None if conversion failed
    """
    # Import here to avoid circular imports
    from decoder import convert_to_audio as orpheus_convert_to_audio

    return orpheus_convert_to_audio(multiframe, count)


async def tokens_decoder(token_gen: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """
    Asynchronous token decoder that converts token stream to audio stream.

    Args:
        token_gen: Asynchronous generator of token strings

    Yields:
        Audio data chunks as bytes
    """
    buffer: list[int] = []
    count: int = 0
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


def tokens_decoder_sync(
    syn_token_gen: Iterator[str], output_file: Optional[Path] = None
) -> list[bytes]:
    """
    Synchronous wrapper for the asynchronous token decoder.

    Args:
        syn_token_gen: Synchronous generator of token strings
        output_file: Path to output WAV file (optional)

    Returns:
        List of audio segments as bytes
    """
    audio_queue: queue.Queue = queue.Queue()
    audio_segments: list[bytes] = []

    # If output_file is provided, prepare WAV file
    wav_file = None
    if output_file:
        # Create directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        wav_file = wave.open(str(output_file), "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

    # Convert the synchronous token generator into an async generator
    async def async_token_gen() -> AsyncIterator[str]:
        for token in syn_token_gen:
            yield token

    async def async_producer() -> None:
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel to indicate completion

    def run_async() -> None:
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
    duration = (
        sum([len(segment) // (2 * 1) for segment in audio_segments]) / SAMPLE_RATE
    )
    logger.debug(f"Generated {len(audio_segments)} audio segments")
    logger.debug(f"Generated {duration:.2f} seconds of audio")

    return audio_segments


def stream_audio(audio_buffer: bytes) -> None:
    """
    Stream audio buffer to output device.

    Args:
        audio_buffer: Audio data as bytes
    """
    if audio_buffer is None or len(audio_buffer) == 0:
        return

    # Convert bytes to NumPy array (16-bit PCM)
    audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

    # Normalize to float in range [-1, 1] for playback
    audio_float = audio_data.astype(np.float32) / 32767.0

    # Play the audio
    sd.play(audio_float, SAMPLE_RATE)
    sd.wait()


def generate_speech_from_api(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    output_file: Optional[Path] = None,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
    chunk_max_length: int = CHUNK_LIMIT,
) -> list[bytes]:
    """
    Generate speech from text using Orpheus model via LM Studio API.

    Args:
        prompt: The text to convert to speech
        voice: The voice to use
        output_file: Path to output WAV file (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty
        chunk_max_length: Maximum length of text chunks for processing

    Returns:
        List of audio segments as bytes
    """

    # If prompt is longer than chunk_max_length, split it into chunks
    if len(prompt) > chunk_max_length:
        chunks = chunk_text(prompt, chunk_max_length)
        all_audio_segments: list[bytes] = []

        logger.info(f"Text split into {len(chunks)} chunks for processing")

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")
            chunk_segments = tokens_decoder_sync(
                generate_tokens_from_api(
                    prompt=chunk,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                ),
                output_file=None,  # Don't write intermediate chunks to file
            )
            all_audio_segments.extend(chunk_segments)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_file), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                for segment in all_audio_segments:
                    wav_file.writeframes(segment)

        duration = (
            sum([len(segment) // (2 * 1) for segment in all_audio_segments])
            / SAMPLE_RATE
        )
        logger.info(f"Generated {len(all_audio_segments)} audio segments")
        logger.info(f"Generated {duration:.2f} seconds of audio")

        return all_audio_segments
    else:
        return tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
            ),
            output_file=output_file,
        )


def list_available_voices() -> None:
    """List all available voices with the recommended one marked."""
    logger.info("Available voices (in order of conversational realism):")
    for voice in AVAILABLE_VOICES:
        marker = "★" if voice == DEFAULT_VOICE else " "
        logger.info(f"{marker} {voice}")
    logger.info(f"\nDefault voice: {DEFAULT_VOICE}")

    logger.info("\nAvailable emotion tags:")
    logger.info(
        "<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
    )


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Args:
        level: The logging level to use
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("orpheus_tts.log"),
        ],
    )


def main() -> None:
    """Main function to handle CLI arguments and execute TTS functionality."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Orpheus Text-to-Speech using LM Studio API"
    )
    parser.add_argument(
        "--text", type=str, help="Text to convert to speech", default=None
    )
    parser.add_argument(
        "--file", type=str, help="Text file to convert to speech", default=None
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Voice to use (default: {DEFAULT_VOICE})",
    )
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument(
        "--list-voices", action="store_true", help="List available voices"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top_p", type=float, default=TOP_P, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=REPETITION_PENALTY,
        help="Repetition penalty (>=1.1 required for stable generation)",
    )
    parser.add_argument(
        "--chunk-max-length",
        type=int,
        default=CHUNK_LIMIT,
        help="Maximum length of text chunks for processing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Set up logging with appropriate level
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    if args.list_voices:
        list_available_voices()
        return

    # Get the text to synthesize
    prompt: Optional[str] = None
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return
        prompt = file_path.read_text(encoding="utf-8")
    elif args.text:
        prompt = args.text
    else:
        # Use text from command line or prompt user
        if len(sys.argv) > 1 and sys.argv[1] not in (
            "--voice",
            "--output",
            "--temperature",
            "--top_p",
            "--repetition_penalty",
            "--debug",
        ):
            prompt = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        else:
            prompt = input("Enter text to synthesize: ")
            if not prompt:
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."

    # Default output file if none provided
    output_file: Optional[Path] = None
    if args.output:
        output_file = Path(args.output)
    else:
        # Create outputs directory if it doesn't exist
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        # Generate a filename based on the voice and a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = outputs_dir / f"{args.voice}_{timestamp}.wav"
        logger.info(f"No output file specified. Saving to {output_file}")

    # Create configuration
    config = TTSConfig(
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        chunk_max_length=args.chunk_max_length,
    )

    # Generate speech
    start_time = time.time()
    if prompt:
        audio_segments = generate_speech_from_api(
            prompt=prompt,
            voice=config.voice,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            output_file=output_file,
            chunk_max_length=config.chunk_max_length,
        )
    end_time = time.time()

    logger.info(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Audio saved to {output_file}")


if __name__ == "__main__":
    main()
