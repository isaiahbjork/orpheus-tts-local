# Orpheus-TTS-Local

A lightweight client for running [Orpheus TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) locally using OpenAI compatible API (e.g. those served via LM Studio, Llama.cpp/Ollama, or OpenWebUI).

## Features

- 🎧 High-quality Text-to-Speech using the Orpheus TTS model
- 💻 Completely local - no cloud API keys needed
- 🔊 Multiple voice options (tara, leah, jess, leo, dan, mia, zac, zoe)
- 💾 Save audio to WAV files

## Quick Setup

1. Install [LM Studio](https://lmstudio.ai/), [Ollama](https://https://ollama.com/download), or [Llama.cpp's Server](https://github.com/ggml-org/llama.cpp)
2. Download the [Orpheus TTS model (orpheus-3b-0.1-ft-q4_k_m.gguf)](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) in your inference engine of choice
  - For Ollama, do `ollama pull hf.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF`
  - For llama.cpp, do `llama-server -hf isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF`
3. Start up the model For LM Studio
  - Load the Orpheus model in LM Studio
  - Start the local server in LM Studio (default: http://127.0.0.1:1234)
4. Install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
5. (For OpenAI compatible endpoint users = Non-LM Studio) Set environment variables as needed
  - export API_PATH=<your_server_url> (e.g. for Ollama, use http://localhost:11434/v1/chat/completions, for OpenWebUI, use http://your_server_url/api/chat/completions)
  - export API_KEY=<your_api_key> (if you've set an API key to be required)
  - export MODEL_NAME=<your_model_name> (if you've set a custom model name in your inference engine)
5. Run the script:
   ```
   python gguf_orpheus.py --text "Hello, this is a test" --voice tara
   ```

## Usage

```
python gguf_orpheus.py --text "Your text here" --voice tara --output "output.wav"
```

### Options

- `--text`: The text to convert to speech
- `--voice`: The voice to use (default: tara)
- `--output`: Output WAV file path (default: auto-generated filename)
- `--list-voices`: Show available voices
- `--temperature`: Temperature for generation (default: 0.6)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--repetition_penalty`: Repetition penalty (default: 1.1)

## Available Voices

- tara - Best overall voice for general use (default)
- leah
- jess
- leo
- dan
- mia
- zac
- zoe

## Emotion
You can add emotion to the speech by adding the following tags:
```xml
<giggle>
<laugh>
<chuckle>
<sigh>
<cough>
<sniffle>
<groan>
<yawn>
<gasp>
```

## License

Apache 2.0

