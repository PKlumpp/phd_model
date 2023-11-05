"""
This is an example how to load the model from Huggingface and use it to
    - Recognize IPA phones
    - Extract CNN features
    - Extract Transformer Encoder features
"""
from decoder.ctc_decoder import decode_lattice
from phonetics.ipa import symbol_to_descriptor, to_symbol
from model.wav2vec2 import Wav2Vec2
from torchinfo import summary
import torch
import numpy as np


def main():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from Huggingface hub
    wav2vec2 = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
    wav2vec2.to(device)
    wav2vec2.eval()

    # Print model summary for batch_size 1 and a single second of audio samples
    summary(wav2vec2, input_size=(1, 16_000), depth=8, device=device)

    # Create new random audio (you can load your own audio here to get actual predictions)
    rand_audio = np.random.rand(1, 16_000)
    
    # IMPORTANT: Always standardize input audio
    mean = rand_audio.mean()
    std = rand_audio.std()
    rand_audio = (rand_audio - mean) / (std + 1e-9)
    
    # Create torch tensor, move to device and feed the model
    rand_audio = torch.tensor(
        rand_audio,
        dtype=torch.float,
        device=device,
    )
    with torch.no_grad():
        y_pred, enc_features, cnn_features = wav2vec2(rand_audio)

    # Decode CTC output for first sample in batch
    phone_sequence, enc_feats, cnn_feats, probs = decode_lattice(
        lattice=y_pred[0].cpu().numpy(),
        enc_feats=enc_features[0].cpu().numpy(),
        cnn_feats=cnn_features[0].cpu().numpy(),
    )
    # phone_sequence contains indices right now. Convert to actual IPA symbols
    symbol_sequence = [to_symbol(i) for i in phone_sequence]

    # Example to convert [œ] to the descriptor "front open-mid rounded vowel"
    print(symbol_to_descriptor("œ"))


if __name__ == "__main__":
    main()
