# Orpheus-TTS-Local

A lightweight client for running [Orpheus TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) locally using LM Studio API over WebRTC with [FastRTC](https://fastrtc.org).

## Features

- 🎧 High-quality Text-to-Speech using the Orpheus TTS model
- 💻 Completely local - no cloud API keys needed
- 🔊 Streaming audio over WebRTC with [FastRTC](https://fastrtc.org)
- 💬 ChatUI for displaying conversation history + downloading audio
- 🔊 Multiple voice options (tara, leah, jess, leo, dan, mia, zac, zoe)

## Quick Setup

1. Install [LM Studio](https://lmstudio.ai/) 
2. Download the [Orpheus TTS model (orpheus-3b-0.1-ft-q4_k_m.gguf)](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) in LM Studio
3. Load the Orpheus model in LM Studio
4. Start the local server in LM Studio (default: http://127.0.0.1:1234)
5. Install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
6. Run the script:
   ```
   python streaming_text_to_speech.py
   ```

## Usage

```
python gguf_orpheus.py --text "Your text here" --voice tara --output "output.wav"
```


## License

Apache 2.0

