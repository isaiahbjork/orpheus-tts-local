from scipy.signal import resample  # Add this import
from snac import SNAC
import numpy as np
import torch
from scipy.io import wavfile  # For loading WAV files
# Initialize the SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {snac_device}")
model = model.to(snac_device)
def encode_audio(audio_bytes, count):
    # Convert audio bytes to tensor and normalize
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_np.astype(np.float32) / 32767.0
    audio_tensor = torch.tensor(audio_float, device=snac_device).unsqueeze(0).unsqueeze(0)
    # Encode audio to get codes
    with torch.inference_mode():
        codes = model.encode(audio_tensor)
    
    # Extract codes for each layer
    codes_0 = codes[0].squeeze(0).cpu().numpy().astype(int)
    codes_1 = codes[1].squeeze(0).cpu().numpy().astype(int)
    codes_2 = codes[2].squeeze(0).cpu().numpy().astype(int)
    T = codes_0.shape[0]
    multiframe = []
    # Interleave codes into multiframe structure
    for j in range(T):
        multiframe.extend([
            codes_0[j],
            codes_1[2*j],
            codes_2[4*j],
            codes_2[4*j + 1],
            codes_1[2*j + 1],
            codes_2[4*j + 2],
            codes_2[4*j + 3],
        ])
    # Generate token strings with proper encoding
    token_strings = []
    for token in multiframe:
        if 0 <= token <= 4096:
            token_value = token + 10 + (count % 7) * 4096
            token_str = f"<custom_token_{token_value}>"
            token_strings.append(token_str)
            count += 1
        else:
            # Handle invalid tokens (skip or log error)
            count += 1  # Increment count to maintain alignment
    return token_strings, count
def audio_encoder(audio_chunk_generator):
    count = 0
    for audio_bytes in audio_chunk_generator:
        token_strings, count = encode_audio(audio_bytes, count)
        for token_str in token_strings:
            yield token_str

def load_wav(file_path, chunk_size=2048):
    sample_rate, audio_data = wavfile.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Convert stereo to mono

    # Resample to 24kHz if necessary
    target_sr = 24000
    if sample_rate != target_sr:
        num_samples = audio_data.shape[0]
        duration = num_samples / sample_rate
        new_num_samples = int(duration * target_sr)
        audio_data = resample(audio_data, new_num_samples).astype(np.int16)  # Resample and convert to int16
        sample_rate = target_sr

    # Ensure the audio is in int16 format (handle edge cases)
    if audio_data.dtype != np.int16:
        if np.issubdtype(audio_data.dtype, np.floating):
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    # Split into chunks
    num_chunks = len(audio_data) // chunk_size
    for i in range(num_chunks):
        chunk = audio_data[i * chunk_size : (i + 1) * chunk_size]
        yield chunk.tobytes()
# 
# Example usage:
if __name__ == "__main__":
    wav_file_path = "sample.wav"
    # Load the WAV file and encode it into tokens
    audio_chunk_generator = load_wav(wav_file_path)
    for token in audio_encoder(audio_chunk_generator):
        print(token)

