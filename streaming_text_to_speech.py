import importlib.util

if importlib.util.find_spec("fastrtc") is None:
    raise RuntimeError("fastrtc is not installed. Please install it using 'pip install fastrtc>=0.0.17'.")

import asyncio
import json
import random

import gradio as gr
import httpx
import numpy as np
import numpy.typing as npt
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    AudioEmitType,
    Stream,
    async_aggregate_bytes_to_16bit,
    wait_for_item,
)
from fastrtc.utils import create_message
from huggingface_hub import InferenceClient

from gguf_orpheus import (
    API_URL,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    HEADERS,
    MAX_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
    format_prompt,
    tokens_decoder,
)
import datetime
async_client = httpx.AsyncClient()

client = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct")

def generate_message():
    system_prompt = """You are a creative text generator that generates short sentences from everyday life.
Example: "Hello!  I'm so excited to talk to you! This is going to be fun!"
Example: I'm nervous about the interview tomorrow
"""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Give me a short sentence please."}
        ],
        max_tokens=100,
        seed=random.randint(0, 1000000)
    )
    msg = response.choices[0].message.content
    if msg:
        msg = msg.replace('"', '')
    return msg


async def generate_tokens_from_api_async(prompt, voice=DEFAULT_VOICE, temperature=TEMPERATURE, 
                            top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    """Generate tokens from text using LM Studio API."""
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    # Create the request payload for the LM Studio API
    payload = {
        "model": "orpheus-3b-0.1-ft-q4_k_m",  # Model name can be anything, LM Studio ignores it
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }

    # Process the streamed response
    token_counter = 0
    async with async_client.stream("POST", API_URL, headers=HEADERS, json=payload) as response:
        async for line in response.aiter_lines():
            if line:
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove the 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        token_text = data['choices'][0].get('text', '')
                        token_counter += 1
                        if token_text:
                            yield token_text
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue
    
    print("Token generation complete")


class OrpheusStream(AsyncStreamHandler):
    def __init__(self):
        super().__init__(output_sample_rate=24_000, output_frame_size=480)
        self.latest_msg = ""
        self.latest_voice_id = DEFAULT_VOICE
        self.audio_queue: asyncio.Queue[AudioEmitType] = asyncio.Queue()
        self.first_chunk = True
    
    async def start_up(self):
        await self.wait_for_args()
        self.start_time = datetime.datetime.now()

    async def receive(self, frame: tuple[int, npt.NDArray[np.int16]]) -> None:
        msg, cb, voice_id, _ = self.latest_args[1:]
        if msg != self.latest_msg or voice_id != self.latest_voice_id:
            await self.send_message(create_message("log", "pause_detected"))
            tokens = generate_tokens_from_api_async(msg, voice_id)
            all_audio = np.array([], dtype=np.int16)
            first_chunk = True
            async for chunk in async_aggregate_bytes_to_16bit(tokens_decoder(tokens)):
                all_audio = np.concatenate([all_audio, chunk.squeeze()])
                if first_chunk:
                    first_chunk = False
                    await self.send_message(create_message("log", "response_starting"))
                print("put chunk at time", datetime.datetime.now())
                await self.audio_queue.put((24_000, chunk))

            cb.append({"role": "user", "content": msg})
            cb.append({"role": "assistant", "content": gr.Audio(value=(24_000, all_audio))})
            await self.audio_queue.put(AdditionalOutputs(cb))
            self.latest_msg = msg
            self.latest_voice_id = voice_id
            

    async def emit(self) -> AudioEmitType:
        item = await wait_for_item(self.audio_queue)
        if item:
            print("got chunk at time", datetime.datetime.now())
        return item

    def copy(self):
        return OrpheusStream()

chat = gr.Chatbot(label="Conversation", type="messages",
                  allow_tags=["giggle", "laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp"])
generate = gr.Button(value="Generate Prompt",)
prompt = gr.Textbox(label="Prompt", value="Hello, how are you?")
stream = Stream(OrpheusStream(), modality="audio", mode="send-receive",
                additional_inputs=[prompt,
                                  chat,
                                  gr.Dropdown(choices=AVAILABLE_VOICES, value=DEFAULT_VOICE, label="Voice"),
                                  generate],
                additional_outputs=[chat],
                additional_outputs_handler=lambda old, new: new, 
                ui_args={"title": "Orpheus TTS WebRTC Streaming",
                         "subtitle": "Powered by FastRTC ⚡️", "send_input_on": "submit"})
with stream.ui:
    generate.click(generate_message, inputs=[], outputs=[prompt])


stream.ui.launch()
