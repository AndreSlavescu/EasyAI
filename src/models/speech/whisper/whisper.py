"""
Implementation of Whisper in torch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoTokenizer

# download tools
from download_audio_sample import download_random_audio_sample

# typing
from typing import Optional

# Whisper vocab size
WHISPER_VOCAB_SIZE = 51864

class LayerNorm(nn.LayerNorm):
    """
    LayerNorm with optional bias. Whisper uses LayerNorm without bias.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=False):
        super().__init__(normalized_shape, eps, elementwise_affine)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model, bias=True)
        self.key_proj = nn.Linear(d_model, d_model, bias=True)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        # qkv: [seq_len, batch_size, d_model]
        seq_len, batch_size, d_model = query.size()

        q = self.query_proj(query).view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = self.key_proj(key).view(-1, batch_size, self.num_heads, self.head_dim)
        v = self.value_proj(value).view(-1, batch_size, self.num_heads, self.head_dim)

        # q^T, k^T, v^T [batch_size * num_heads, seq_len, head_dim]
        q = q.permute(1, 2, 0, 3).reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        k = k.permute(1, 2, 0, 3).reshape(batch_size * self.num_heads, -1, self.head_dim)
        v = v.permute(1, 2, 0, 3).reshape(batch_size * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_weights += attn_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(seq_len, batch_size, d_model)

        output = self.out_proj(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn, bias=True)
        self.fc2 = nn.Linear(d_ffn, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output = self.self_attn(x, x, x, attn_mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        attn_output = self.cross_attn(x, encoder_output, encoder_output, memory_mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm2(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm3(x)
        return x

class WhisperEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ffn, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.layer_norm(x)
        return x

class WhisperDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ffn, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        x = self.layer_norm(x)
        return x

class Whisper(nn.Module):
    def __init__(
            self,
            vocab_size: int = WHISPER_VOCAB_SIZE,
            d_model: int = 512,
            encoder_layers: int = 6,
            decoder_layers: int = 6,
            num_heads: int = 8,
            d_ffn: int = 2048,
            dropout: float = 0.1,
            num_mel_bins: int = 80
        ):
        super().__init__()
        self.mel_spectrogram = MelSpectrogram(num_mel_filterbanks=num_mel_bins)
        self.audio_projection = nn.Linear(num_mel_bins, d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = WhisperEncoder(
            num_layers=encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ffn=d_ffn,
            dropout=dropout
        )
        self.decoder = WhisperDecoder(
            num_layers=decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ffn=d_ffn,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, audio_waveform, decoder_input_ids, tgt_mask=None):
        mel_spec = self.mel_spectrogram(audio_waveform)  # [batch_size, n_mels, time]
        mel_spec = mel_spec.permute(2, 0, 1)  # [time, batch_size, n_mels]
        encoder_input = self.audio_projection(mel_spec)  # [time, batch_size, d_model]
        encoder_input = self.positional_encoding(encoder_input)
        encoder_output = self.encoder(encoder_input)  # [time, batch_size, d_model]
        text_embedded = self.text_embedding(decoder_input_ids).permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        text_embedded = self.positional_encoding(text_embedded)
        decoder_output = self.decoder(
            text_embedded,
            encoder_output,
            tgt_mask=tgt_mask
        )  # [seq_len, batch_size, d_model]
        logits = self.fc_out(decoder_output).permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
        return logits

    def generate(self, audio_waveform, max_length=448, device='cpu'):
        self.eval()
        with torch.no_grad():
            mel_spec = self.mel_spectrogram(audio_waveform.to(device))  # [batch_size, n_mels, time]
            mel_spec = mel_spec.permute(2, 0, 1)  # [time, batch_size, n_mels]
            encoder_input = self.audio_projection(mel_spec)  # [time, batch_size, d_model]
            encoder_input = self.positional_encoding(encoder_input)
            encoder_output = self.encoder(encoder_input)  # [time, batch_size, d_model]

            decoder_input_ids = torch.tensor([[50257]], device=device)
            generated_ids = []

            for _ in range(max_length):
                text_embedded = self.text_embedding(decoder_input_ids).permute(1, 0, 2)  # [seq_len, batch_size, d_model]
                text_embedded = self.positional_encoding(text_embedded)
                decoder_output = self.decoder(
                    text_embedded,
                    encoder_output
                )  # [seq_len, batch_size, d_model]
                logits = self.fc_out(decoder_output[-1]).unsqueeze(0)  # [1, batch_size, vocab_size]
                next_token_id = logits.argmax(dim=-1)  # [1, batch_size]
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.transpose(0, 1)], dim=1)
                generated_ids.append(next_token_id.item())
                if next_token_id.item() == 50256:
                    break
            return decoder_input_ids[:, 1:]

class MelSpectrogram(nn.Module):
    def __init__(
            self,
            sample_rate: int = 16000,
            num_fft: int = 400,
            window_length: int = 400,
            stft_window_hop_length: int = 160,
            num_mel_filterbanks: int = 80,
            f_min: float = 0.0,
            f_max: Optional[float] = None
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = num_fft
        self.w_length = window_length
        self.hop = stft_window_hop_length
        self.n_mels = num_mel_filterbanks
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2  # nyquist frequency (freq_samples * 1/2)
        self.window = torch.hann_window(window_length)
        self.mel_filterbank = self.create_mel_filterbank(
            self.f_min, self.f_max, self.n_mels, self.n_fft, self.sample_rate
        )

    @staticmethod
    def create_mel_filterbank(f_min, f_max, n_mels, n_fft, sample_rate):
        def hz_to_mel(hz): return 2595.0 * torch.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(torch.tensor(f_min))
        mel_max = hz_to_mel(torch.tensor(f_max))
        mels = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz = mel_to_hz(mels)
        bins = torch.floor((n_fft + 1) * hz / sample_rate).long()

        filterbank = torch.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            lower = bins[i - 1].item()
            center = bins[i].item()
            upper = bins[i + 1].item()

            if center > lower:
                filterbank[i - 1, lower:center] = (
                    torch.arange(lower, center) - lower
                ) / (center - lower)
            if upper > center:
                filterbank[i - 1, center:upper] = (
                    upper - torch.arange(center, upper)
                ) / (upper - center)

        return filterbank

    def stft(self, waveform):
        return torch.stft(
            waveform, self.n_fft, self.hop, self.w_length,
            self.window.to(waveform.device), return_complex=True
        )

    def mel_spectrogram(self, power_spec):
        mel_filterbank = self.mel_filterbank.to(power_spec.device)
        return torch.matmul(mel_filterbank, power_spec)

    def forward(self, waveform):
        eps = 1e-10
        stft_out = self.stft(waveform)
        power_spec = stft_out.abs() ** 2
        mel_spec = self.mel_spectrogram(power_spec)
        log_mel_spec = torch.log10(mel_spec + eps)
        return log_mel_spec  # [batch_size, n_mels, time]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)].unsqueeze(1)
        return self.dropout(x)

"""
Testing Whisper with random audio
"""
def test():
    model = Whisper(
        vocab_size=WHISPER_VOCAB_SIZE,
        d_model=512,
        encoder_layers=6,
        decoder_layers=6,
        num_heads=8,
        d_ffn=2048,
        dropout=0.1,
        num_mel_bins=80
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    audio_path = download_random_audio_sample()

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        waveform = resampler(waveform)

    decoder_input_ids = torch.tensor([[50257]], device=device)
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-small")
    generated_ids = []
    max_length = 50

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(waveform, decoder_input_ids)
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True) 
            generated_ids.append(next_token_id.item())
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1) 

            if next_token_id.item() == 50256: 
                break

    transcription = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n\n##### Audio sample transcription #####\n\n{transcription}")

if __name__ == "__main__":
    test()