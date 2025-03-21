# Orpheus-TTS-Local

A lightweight client for running [Orpheus TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) locally using LM Studio API.

## Features

- 🎧 High-quality Text-to-Speech using the Orpheus TTS model
- 💻 Completely local - no cloud API keys needed
- 🔊 Multiple voice options (tara, leah, jess, leo, dan, mia, zac, zoe)
- 💾 Save audio to WAV files

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
   python gguf_orpheus.py --text "Hello, this is a test" --voice tara
   ```

## Usage

### Monologue

```
python gguf_orpheus.py --text "Your text here" --voice tara --output "output.wav"
```

#### Options

- `--text`: The text to convert to speech
- `--voice`: The voice to use (default: tara)
- `--output`: Output WAV file path (default: auto-generated filename)
- `--list-voices`: Show available voices
- `--temperature`: Temperature for generation (default: 0.6)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--repetition_penalty`: Repetition penalty (default: 1.1)

### Dialogue

```
python dialogue.py --input-text 'leo: That makes sense. Thank you for your help, Doctor Leah. \
leah: You’re welcome, Leo. Take care and feel better soon!'
python dialogue.py --input-file ./dialogue_example.txt
```

#### Options

- `--input-text`: The text to convert to speech
- `--input-file`: The text file to convert to speech
- `--output`: Output WAV file path (default: auto-generated filename)
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
