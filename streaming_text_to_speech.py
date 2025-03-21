from gguf_orpheus import generate_tokens_from_api, tokens_decoder, AVAILABLE_VOICES, DEFAULT_VOICE
from fastrtc import Stream, AsyncStreamHandler, AudioEmitType, wait_for_item, async_aggregate_bytes_to_16bit, AdditionalOutputs
from fastrtc.utils import create_message
from asyncio import Queue
import gradio as gr
import numpy.typing as npt
import numpy as np
import asyncio

class OrpheusStream(AsyncStreamHandler):
    def __init__(self):
        super().__init__(output_sample_rate=24_000, output_frame_size=480)
        self.latest_msg = ""
        self.latest_voice_id = DEFAULT_VOICE
        self.audio_queue: Queue[AudioEmitType] = Queue()
    
    async def start_up(self):
        await self.wait_for_args()

    async def receive(self, frame: tuple[int, npt.NDArray[np.int16]]) -> None:
        msg, cb, voice_id = self.latest_args[1:]
        if msg != self.latest_msg or voice_id != self.latest_voice_id:
            await self.send_message(create_message("log", "pause_detected"))
            await asyncio.sleep(0.05)
            tokens = generate_tokens_from_api(msg, voice_id)
            async def async_iterate_tokens(tokens):
                for token in tokens:
                    yield token
            audio_chunks = []
            async for chunk in async_aggregate_bytes_to_16bit(tokens_decoder(async_iterate_tokens(tokens))):
                await self.audio_queue.put((24_000, chunk))
                audio_chunks.append(chunk.squeeze())
                if len(audio_chunks) == 1:
                    await self.send_message(create_message("log", "response_starting"))
            all_audio = np.concatenate(audio_chunks)
            cb.append({"role": "user", "content": msg})
            cb.append({"role": "assistant", "content": gr.Audio(value=(24_000, all_audio))})
            await self.audio_queue.put(AdditionalOutputs(cb))
            self.latest_msg = msg
            self.latest_voice_id = voice_id

    async def emit(self) -> AudioEmitType:
        return await wait_for_item(self.audio_queue)

    def copy(self):
        return OrpheusStream()

chat = gr.Chatbot(label="Conversation", type="messages")
stream = Stream(OrpheusStream(), modality="audio", mode="send-receive",
                additional_inputs=[gr.Textbox(label="Prompt", value="Hello, how are you?"),
                                  chat,
                                gr.Dropdown(choices=AVAILABLE_VOICES, value=DEFAULT_VOICE, label="Voice")],
                additional_outputs=[chat],
                additional_outputs_handler=lambda old, new: new, 
                ui_args={"title": "Orpheus TTS WebRTC Streaming",
                         "subtitle": "Powered by FastRTC ⚡️", "send_input_on": "submit"})
stream.ui.launch()
