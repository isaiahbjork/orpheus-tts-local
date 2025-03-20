import base64
import io
import time

import numpy as np
import soundfile as sf  # type: ignore
import streamlit as st

from gguf_orpheus import (
    API_URL as DEFAULT_API_URL,
)

# Import from the Orpheus script
from gguf_orpheus import (
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    REPETITION_PENALTY,
    SAMPLE_RATE,
    TEMPERATURE,
    TOP_P,
    generate_speech_from_api,
)
from gguf_orpheus import (
    HEADERS as DEFAULT_HEADERS,
)


def get_audio_bytes(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    Create an HTML audio player for the given audio data.

    Args:
        audio_data: Audio data as a numpy array
        sample_rate: Sample rate of the audio data

    """
    # Convert to bytes
    virtual_file = io.BytesIO()
    sf.write(virtual_file, audio_data, sample_rate, format="WAV")
    audio_bytes = virtual_file.getvalue()

    return audio_bytes


def get_download_link(
    audio_data: np.ndarray, filename: str, sample_rate: int = SAMPLE_RATE
) -> str:
    """
    Generate a download link for audio data.

    Args:
        audio_data: Audio data as a numpy array
        filename: Desired filename for download
        sample_rate: Sample rate of the audio data

    Returns:
        HTML string with download link
    """
    # Convert to bytes
    virtual_file = io.BytesIO()
    sf.write(virtual_file, audio_data, sample_rate, format="WAV")
    audio_bytes = virtual_file.getvalue()

    # Create download link
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


def combined_audio_segments(audio_segments: list[bytes]) -> np.ndarray:
    """
    Combine multiple audio segments into a single numpy array.

    Args:
        audio_segments: List of audio segment bytes

    Returns:
        Combined audio as numpy array
    """
    # Convert bytes to numpy arrays
    audio_arrays = [
        np.frombuffer(segment, dtype=np.int16) for segment in audio_segments
    ]

    # Combine arrays
    combined = np.concatenate(audio_arrays)

    # Convert to float in range [-1, 1] for processing
    return combined.astype(np.float32) / 32767.0


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Orpheus TTS", page_icon="🎙️", layout="wide")

    # Initialize session state for connection status if not exists
    if "connection_active" not in st.session_state:
        st.session_state.connection_active = None
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = []

    st.title("Orpheus Text-to-Speech")
    st.markdown("""
    This app uses the Orpheus TTS model to generate realistic speech with emotions.
    Make sure you have LM Studio running with the Orpheus model loaded.
    """)

    st.divider()

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Connection settings
    with st.sidebar.expander("Connection Settings", expanded=False):
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        api_key = st.text_input("API Key (if needed)", value="", type="password")
        st.caption("Leave API Key empty for local LM Studio without authentication")

        # Add connection status check in sidebar
        if st.button("Test Connection", key="sidebar_test_conn"):
            try:
                import requests

                # Prepare headers
                headers = DEFAULT_HEADERS.copy()
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                # Simple test request
                test_payload = {"prompt": "test", "max_tokens": 1, "stream": False}

                with st.spinner("Testing..."):
                    response = requests.post(
                        api_url, headers=headers, json=test_payload, timeout=5
                    )

                if response.status_code == 200:
                    st.sidebar.success(f"✅ Connected to {api_url.split('/')[-3]}")
                    # Store connection status in session state
                    st.session_state.connection_active = True
                else:
                    st.sidebar.error(f"❌ Failed: Status {response.status_code}")
                    st.session_state.connection_active = False
            except Exception as e:
                error_msg = str(e)
                if "NewConnectionError" in error_msg and "refused" in error_msg:
                    st.sidebar.error("❌ Connection refused. Is LM Studio running?")
                elif "ConnectTimeoutError" in error_msg:
                    st.sidebar.error("❌ Connection timeout. Check server address.")
                else:
                    st.sidebar.error(f"❌ Error: {error_msg[:50]}...")
                st.session_state.connection_active = False

    # Display connection status indicator in sidebar
    status_col1, _ = st.sidebar.columns([1, 4])
    with status_col1:
        if st.session_state.connection_active is None:
            st.sidebar.caption("⚪ Connection status unknown")
        elif st.session_state.connection_active is True:
            st.sidebar.caption("🟢 Connection active")
        else:
            st.sidebar.caption("🔴 Connection error")

    # Voice selection with tooltips
    voice_descriptions = {
        "tara": "Best overall voice",
        "leah": "",
        "jess": "",
        "leo": "",
        "dan": "",
        "mia": "",
        "zac": "",
        "zoe": "",
    }

    selected_voice = st.sidebar.selectbox(
        "Select Voice",
        AVAILABLE_VOICES,
        index=AVAILABLE_VOICES.index(DEFAULT_VOICE),
        format_func=lambda x: f"{x} - {voice_descriptions[x]}"
        if voice_descriptions[x]
        else x,
    )

    # Advanced options
    with st.sidebar.expander("Options"):
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            TEMPERATURE,
            0.05,
            help="Higher values make speech more creative but less stable",
        )
        top_p = st.slider(
            "Top P", 0.1, 1.0, TOP_P, 0.05, help="Controls diversity of word choices"
        )
        repetition_penalty = st.slider(
            "Repetition Penalty",
            1.0,
            2.0,
            REPETITION_PENALTY,
            0.1,
            help="Prevents repetition of phrases, values >=1.1 recommended",
        )
        max_tokens = st.slider(
            label="Max Tokens",
            min_value=100,
            max_value=8096,
            value=1200,
            step=100,
            help="Maximum length of generated speech",
        )

    # Emotion tags info
    st.sidebar.header("Emotion Tags")
    st.sidebar.markdown("""
    Insert these tags in your text for emotional speech:
    - `<laugh>` - Laughter
    - `<chuckle>` - Soft laughter
    - `<sigh>` - Sigh
    - `<cough>` - Cough
    - `<sniffle>` - Sniffle
    - `<groan>` - Groan
    - `<yawn>` - Yawn
    - `<gasp>` - Gasp
    """)

    examples = {
        "Basic greeting": f"Hello, my name is {selected_voice.capitalize()}. I'm a text-to-speech model that can speak with emotions.",
        "Emotional story": "I was so nervous before the presentation <sigh>, but then I remembered all my preparation. When I finished, everyone applauded <laugh> and I felt so relieved!",
        "Technical explanation": "Orpheus TTS is a state-of-the-art, Llama-based Speech-LLM designed for high-quality, empathetic text-to-speech generation. This model is the base model that can be used for many downstream tasks, like TTS, Zero-shot voice cloning and classification.",
    }

    example_prompt = st.selectbox("Example Prompts:", list(examples.keys()))

    # Main input area
    input_text = st.text_area(
        "Enter text to convert to speech", value=examples[example_prompt], height=150
    )

    # Generate button
    if st.button("Generate Speech"):
        if not input_text:
            st.error("Please enter some text to convert to speech.")
        else:
            with st.spinner(show_time=True):
                # Call the generation function
                try:
                    # Prepare API headers
                    headers = DEFAULT_HEADERS.copy()
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"

                    # Monkey patch the API_URL and HEADERS in the module
                    import gguf_orpheus

                    original_api_url = gguf_orpheus.API_URL
                    original_headers = gguf_orpheus.HEADERS
                    gguf_orpheus.API_URL = api_url
                    gguf_orpheus.HEADERS = headers

                    try:
                        audio_segments = generate_speech_from_api(
                            prompt=input_text,
                            voice=selected_voice,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            repetition_penalty=repetition_penalty,
                        )
                    finally:
                        # Restore original values
                        gguf_orpheus.API_URL = original_api_url
                        gguf_orpheus.HEADERS = original_headers

                    # Convert segments to a single audio array
                    if audio_segments:
                        combined_audio = combined_audio_segments(audio_segments)

                        # Display audio player
                        st.subheader("Generated Speech")
                        st.session_state.last_audio.append(
                            {
                                "audio": get_audio_bytes(combined_audio),
                                "name": f"{selected_voice}_{int(time.time())}",
                                "text": input_text,
                            }
                        )
                        st.session_state.connection_active = True

                    else:
                        st.error(
                            "No audio was generated. Check if LM Studio is running with the Orpheus model loaded."
                        )

                except Exception as e:
                    error_msg = str(e)

                    # Show user-friendly error message in sidebar
                    if "NewConnectionError" in error_msg and "refused" in error_msg:
                        st.sidebar.error("❌ Connection refused")
                        error_details = "The server actively refused the connection. Make sure LM Studio is running."
                    elif "ConnectTimeoutError" in error_msg:
                        st.sidebar.error("❌ Connection timeout")
                        error_details = "The connection timed out. Check if the server address is correct."
                    elif "Max retries exceeded" in error_msg:
                        st.sidebar.error("❌ Max retries exceeded")
                        error_details = "Could not connect after multiple attempts. Is the server running?"
                    else:
                        st.sidebar.error("❌ Connection error")
                        error_details = error_msg

                    # Update session state
                    st.session_state.connection_active = False

                    # Show detailed error in main area
                    st.error(f"Error generating speech: {error_details}")

                    # Show a more technical error message in an expander for debugging
                    with st.expander("Technical Error Details"):
                        st.code(error_msg)
    if st.session_state.last_audio:
        st.audio(data=st.session_state.last_audio[-1]["audio"], format="audio/wav")

        # Provide download link
        output_filename = st.text_input(
            "Output filename",
            value=f"{selected_voice}_{int(time.time())}.wav",
        )
        # Ensure proper extension
        if not output_filename.endswith(".wav"):
            output_filename += ".wav"
        st.download_button(
            label="download audio",
            file_name=output_filename,
            mime="audio/wav",
            data=st.session_state.last_audio[-1]["name"],
        )
        st.subheader("History")
        for audio_file in reversed(st.session_state.last_audio[:-1]):
            with st.expander(
                label=f"{audio_file['text'][:60]}{'...' if len(audio_file['text']) > 60 else ''}",
                expanded=False,
            ):
                st.audio(data=audio_file["audio"], format="audio/wav")

                # Provide download link
                output_filename = st.text_input(
                    "Output filename",
                    value=audio_file["name"],
                )
                # Ensure proper extension
                if not output_filename.endswith(".wav"):
                    output_filename += ".wav"
                st.download_button(
                    label="download audio",
                    file_name=output_filename,
                    mime="audio/wav",
                    data=audio_file["audio"],
                )


if __name__ == "__main__":
    main()
