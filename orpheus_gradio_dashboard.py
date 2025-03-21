import os
import queue
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Generator, Optional

import gradio as gr
import numpy as np
import sounddevice as sd

# Import from the improved Orpheus TTS client
from gguf_orpheus import (
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    REPETITION_PENALTY,
    SAMPLE_RATE,
    TEMPERATURE,
    TOP_P,
    convert_to_audio,
    generate_tokens_from_api,
    turn_token_into_id,
)

# Emotion tags from the original script
EMOTION_TAGS = {
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
}

# Create a temporary directory for audio files
TEMP_DIR = Path(tempfile.gettempdir()) / "orpheus_tts_gradio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Global flag to control streaming
stop_streaming = False


def clean_old_files() -> None:
    """Clean up old temporary audio files (older than 1 hour)."""
    current_time = time.time()
    for file_path in TEMP_DIR.glob("*.wav"):
        # If file is older than 1 hour, delete it
        if current_time - file_path.stat().st_mtime > 3600:
            try:
                file_path.unlink()
                print(f"Deleted old file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


def stream_real_time(
    tokens_generator: Generator[str, None, None],
    status_queue: queue.Queue,
    output_file: Optional[Path] = None,
) -> None:
    """
    Process tokens in real-time and play audio chunks as they're generated.

    Args:
        tokens_generator: Generator yielding token strings
        status_queue: Queue for status updates
        output_file: Optional path to save the complete audio
    """
    global stop_streaming
    stop_streaming = False

    # Initialize variables
    buffer = []
    count = 0
    all_audio_chunks = []
    wav_file = None

    # Open output file if provided
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        wav_file = wave.open(str(output_file), "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

    # Process tokens in real-time
    for token_text in tokens_generator:
        if stop_streaming:
            status_queue.put("Streaming stopped by user")
            break

        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                try:
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        # Add to our collection
                        all_audio_chunks.append(audio_samples)

                        # Write to WAV file if provided
                        if wav_file:
                            wav_file.writeframes(audio_samples)

                        # Play the audio in real-time
                        audio_data = np.frombuffer(audio_samples, dtype=np.int16)
                        audio_float = audio_data.astype(np.float32) / 32767.0
                        sd.play(audio_float, SAMPLE_RATE)

                        # Update status
                        status_queue.put(
                            f"Processing: Generated {count} tokens, {len(all_audio_chunks)} audio chunks"
                        )
                except Exception as e:
                    status_queue.put(f"Error processing audio: {str(e)}")

    # Close WAV file if opened
    if wav_file:
        wav_file.close()

    # Calculate total duration
    total_duration = sum([len(chunk) for chunk in all_audio_chunks]) / 2 / SAMPLE_RATE
    status_queue.put(f"Complete: Generated {total_duration:.2f} seconds of audio")

    return


def generate_speech_streaming(
    text: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    status_queue: queue.Queue = None,
) -> Path:
    """
    Generate speech from text and stream it in real-time.

    Args:
        text: Text to convert to speech
        voice: Voice to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        status_queue: Queue for status updates

    Returns:
        Path to the saved audio file
    """
    if not text.strip():
        if status_queue:
            status_queue.put("Error: Text cannot be empty")
        return None

    try:
        # Clean up old files before generating new ones
        clean_old_files()

        # Create a timestamped output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = TEMP_DIR / f"{voice}_{timestamp}.wav"

        # Update status
        if status_queue:
            status_queue.put("Starting speech generation...")

        # Generate tokens
        tokens_generator = generate_tokens_from_api(
            prompt=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Stream audio in real-time
        stream_real_time(tokens_generator, status_queue, output_file)

        return output_file

    except Exception as e:
        if status_queue:
            status_queue.put(f"Error: {str(e)}")
        return None


def create_ui() -> gr.Blocks:
    """Create the Gradio UI for the Orpheus TTS system with real-time streaming."""
    global stop_streaming

    with gr.Blocks(title="Orpheus TTS Streaming Dashboard") as ui:
        gr.Markdown("# Orpheus Text-to-Speech Streaming Dashboard")
        gr.Markdown(
            "Generate and hear natural-sounding speech in real-time using Orpheus TTS and LM Studio."
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Input text area
                text_input = gr.Textbox(
                    label="Text to Speak", placeholder="Enter text here...", lines=5
                )

                # Voice selection
                voice_dropdown = gr.Dropdown(
                    choices=AVAILABLE_VOICES, value=DEFAULT_VOICE, label="Voice"
                )

                # Emotion tag examples
                gr.Markdown("### Emotion Tags")
                gr.Markdown(
                    "Add these tags in your text: " + ", ".join(sorted(EMOTION_TAGS))
                )
                gr.Markdown("Example: 'I'm so happy to see you! <laugh>'")

                # Advanced settings (collapsible)
                with gr.Accordion("Advanced Settings", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=TEMPERATURE,
                        step=0.05,
                        label="Temperature",
                    )

                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=TOP_P, step=0.05, label="Top-P"
                    )

                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=REPETITION_PENALTY,
                        step=0.05,
                        label="Repetition Penalty",
                    )

                with gr.Row():
                    # Generate button
                    generate_btn = gr.Button(
                        "Generate and Stream Speech", variant="primary"
                    )

                    # Stop button
                    stop_btn = gr.Button("Stop Streaming", variant="stop")

                # Status message
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                # Audio output (for completed file)
                audio_output = gr.Audio(label="Completed Audio File", type="filepath")

                # Instructions
                gr.Markdown("### Instructions")
                gr.Markdown(
                    """
                    1. Enter text in the left panel
                    2. Select a voice from the dropdown
                    3. Optionally add emotion tags like <laugh> or <sigh>
                    4. Click "Generate and Stream Speech"
                    5. Audio will play immediately as it's generated
                    6. Click "Stop Streaming" to interrupt
                    7. The complete audio file will appear here when finished
                    
                    Make sure LM Studio is running locally with the Orpheus model loaded.
                    
                    ### Real-time Streaming
                    You will hear the audio as it's being generated through your speakers.
                    The complete file will be available for download when finished.
                    """
                )

        # Handle the stop button click
        def on_stop_click():
            global stop_streaming
            stop_streaming = True
            return "Stopping streaming..."

        stop_btn.click(fn=on_stop_click, outputs=[status])

        # Handle the generate button click
        def on_generate_click(text, voice, temperature, top_p, repetition_penalty):
            global stop_streaming
            stop_streaming = False
            status_queue = queue.Queue()

            # Update status immediately
            yield "Initializing speech generation...", None

            # Run text-to-speech in a separate thread
            output_file_queue = queue.Queue()

            def process_streaming():
                output_file = generate_speech_streaming(
                    text=text,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    status_queue=status_queue,
                )
                output_file_queue.put(output_file)

            thread = threading.Thread(target=process_streaming)
            thread.start()

            # Keep updating the status while the thread is running
            while thread.is_alive():
                if not status_queue.empty():
                    message = status_queue.get()
                    yield message, None
                time.sleep(0.1)

            # Get any final messages from the queue
            final_status = "Speech generation complete"
            while not status_queue.empty():
                final_status = status_queue.get()

            # Get the output file from the queue
            output_file = None
            if not output_file_queue.empty():
                output_file = output_file_queue.get()

            if output_file:
                yield final_status, str(output_file)
            else:
                yield "Error: Failed to generate audio file", None

        generate_btn.click(
            fn=on_generate_click,
            inputs=[
                text_input,
                voice_dropdown,
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider,
            ],
            outputs=[status, audio_output],
        )

        # Add examples
        gr.Examples(
            examples=[
                [
                    "Hello, my name is Tara. I'm an AI assistant. <laugh> How can I help you today?",
                    "tara",
                ],
                [
                    "Good morning everyone! <sigh> I didn't get much sleep last night, but I'm ready for our meeting.",
                    "leah",
                ],
                [
                    "The sky is clear, and the moon is bright tonight. <chuckle> It reminds me of a story I once heard.",
                    "leo",
                ],
                [
                    "I can't believe you said that! <laugh> That's the funniest thing I've heard all day!",
                    "zoe",
                ],
            ],
            inputs=[text_input, voice_dropdown],
        )

    return ui


def main():
    """Run the Gradio application."""
    ui = create_ui()
    ui.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
