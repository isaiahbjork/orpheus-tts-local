import io
import logging
import time
from typing import Optional

import numpy as np
import requests
import soundfile as sf  # type: ignore
import streamlit as st

from gguf_orpheus import (
    API_URL as DEFAULT_API_URL,
)
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

# Set up logging for debugging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_audio_bytes(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert numpy audio data to WAV format and return as bytes.

    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.
        sample_rate (int, optional): The sample rate of the audio. Defaults to SAMPLE_RATE.

    Returns:
        bytes: The WAV audio data in bytes format.
    """
    virtual_file = io.BytesIO()
    sf.write(virtual_file, audio_data, sample_rate, format="WAV")
    return virtual_file.getvalue()


def combined_audio_segments(audio_segments: list[bytes]) -> np.ndarray:
    """Combine multiple audio segments (byte arrays) into a single NumPy array.

    Args:
        audio_segments (List[bytes]): List of audio segments in byte format.

    Returns:
        np.ndarray: Combined audio as a NumPy array normalized to float32.
    """
    audio_arrays = [
        np.frombuffer(segment, dtype=np.int16) for segment in audio_segments
    ]
    combined = np.concatenate(audio_arrays)
    return combined.astype(np.float32) / 32767.0


def check_connection(api_url: str, api_key: Optional[str]) -> bool:
    """Check the connection status to the API.

    Args:
        api_url (str): The API URL.
        api_key (Optional[str]): The API key if authentication is required.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    headers = DEFAULT_HEADERS.copy()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(api_url, headers=headers, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="Orpheus TTS", page_icon="🎙️", layout="wide")

    # Initialize session state
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

    # Sidebar settings
    st.sidebar.header("Configuration")

    # Connection settings
    with st.sidebar.expander("Connection Settings", expanded=False):
        api_url = st.text_input(label="API URL", value=DEFAULT_API_URL)
        api_key = st.text_input(label="API Key (if needed)", value="", type="password")

        # Automatically check connection when app starts
        if st.session_state.connection_active is None:
            st.session_state.connection_active = check_connection(
                api_url=api_url, api_key=api_key
            )

        # Manual test connection button
        if st.button("Test Connection"):
            st.session_state.connection_active = check_connection(
                api_url=api_url, api_key=api_key
            )

    # Display connection status
    status_col1, _ = st.sidebar.columns([1, 4])
    with status_col1:
        if st.session_state.connection_active is None:
            st.sidebar.caption("⚪ Connection status unknown")
        elif st.session_state.connection_active:
            st.sidebar.caption("🟢 Connection active")
        else:
            st.sidebar.caption("🔴 Connection error")

    # Voice selection with descriptions
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
        label="Select Voice",
        options=AVAILABLE_VOICES,
        index=AVAILABLE_VOICES.index(DEFAULT_VOICE),
        format_func=lambda x: f"{x} - {voice_descriptions[x]}"
        if voice_descriptions[x]
        else x,
    )

    with st.sidebar.expander("Options"):
        temperature = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=TEMPERATURE,
            step=0.05,
        )
        top_p = st.slider(
            label="Top P", min_value=0.1, max_value=1.0, value=TOP_P, step=0.05
        )
        repetition_penalty = st.slider(
            label="Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=REPETITION_PENALTY,
            step=0.1,
        )
        max_tokens = st.slider(
            label="Max Tokens", min_value=100, max_value=8096, value=2400, step=100
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

    example_prompt = st.selectbox(
        label="Example Prompts:", options=list(examples.keys())
    )
    input_text = st.text_area(
        label="Enter text to convert to speech",
        value=examples[example_prompt],
        height=150,
    )

    # Generate Speech Button
    if st.button(
        "Generate Speech",
        disabled=not st.session_state.connection_active,
        type="primary",
    ):
        if not input_text:
            st.error("Please enter some text.")
        else:
            with st.spinner("Generating speech...", show_time=True):
                try:
                    import gguf_orpheus

                    original_api_url, original_headers = (
                        gguf_orpheus.API_URL,
                        gguf_orpheus.HEADERS,
                    )
                    gguf_orpheus.API_URL, gguf_orpheus.HEADERS = (
                        api_url,
                        DEFAULT_HEADERS.copy(),
                    )

                    if api_key:
                        gguf_orpheus.HEADERS["Authorization"] = f"Bearer {api_key}"

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
                        gguf_orpheus.API_URL, gguf_orpheus.HEADERS = (
                            original_api_url,
                            original_headers,
                        )

                    if audio_segments:
                        combined_audio = combined_audio_segments(audio_segments)
                        audio_bytes = get_audio_bytes(combined_audio)

                        # Store last 20 results
                        st.session_state.last_audio.append(
                            {
                                "audio": audio_bytes,
                                "name": f"{selected_voice}_{int(time.time())}.wav",
                                "text": input_text,
                            }
                        )
                        if len(st.session_state.last_audio) > 20:
                            st.session_state.last_audio.pop(0)

                    else:
                        st.error("No audio generated. Check LM Studio.")

                except Exception as e:
                    logging.error(f"Speech generation error: {e}")
                    st.error("Error generating speech. Check the logs for details.")

    st.divider()

    # Audio Player and Download
    if st.session_state.last_audio:
        last_audio = st.session_state.last_audio[-1]
        st.subheader("Result")
        st.audio(data=last_audio["audio"], format="audio/wav")

        st.download_button(
            label="Download audio",
            file_name=last_audio["name"],
            mime="audio/wav",
            data=last_audio["audio"],
        )

    # History Section (Last 20 Results)
    if len(st.session_state.last_audio) > 1:
        st.subheader("History")
        st.caption(
            "Only the last 20 audio clips are saved. Older clips will be removed automatically."
        )

        for audio_file in reversed(st.session_state.last_audio[:-1]):
            preview_text = audio_file["text"][:60] + (
                "..." if len(audio_file["text"]) > 60 else ""
            )
            with st.expander(label=f"{preview_text}", expanded=False):
                st.audio(data=audio_file["audio"], format="audio/wav")
                st.download_button(
                    label="Download",
                    file_name=audio_file["name"],
                    mime="audio/wav",
                    data=audio_file["audio"],
                )


if __name__ == "__main__":
    main()
