"""Seq2Seq model with attention for handwritten equation recognition."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """CNN encoder for image features."""

    def __init__(self, input_channels=1, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        # After 3 poolings: 640x480 -> 80x60
        self.fc = nn.Linear(64 * 80 * 60, output_dim)

    def forward(self, x):
        # x: (batch, 1, 640, 480)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Attention(nn.Module):
    """Attention mechanism."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, 1, hidden_dim) [single encoder output]
        batch_size = decoder_hidden.size(0)

        # Compute attention scores
        combined = torch.cat([decoder_hidden.unsqueeze(1), encoder_outputs], dim=2)
        attn_weights = torch.softmax(self.attn(combined), dim=1)
        context = (attn_weights * encoder_outputs).sum(dim=1)
        return context, attn_weights


class LSTMDecoder(nn.Module):
    """LSTM decoder with attention."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_token, hidden, cell, encoder_output):
        # input_token: (batch,)
        # hidden, cell: (batch, hidden_dim)
        # encoder_output: (batch, 1, hidden_dim)

        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)

        # Apply attention
        context, attn_weights = self.attention(hidden, encoder_output)

        # Concatenate embedding with context
        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        output = output.squeeze(1)

        # Predict next token
        logits = self.fc(output)

        return logits, hidden.squeeze(0), cell.squeeze(0), attn_weights


class Seq2Seq(nn.Module):
    """Full seq2seq model."""

    def __init__(self, vocab_size, encoder_output_dim=256, decoder_embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(output_dim=encoder_output_dim)
        self.decoder = LSTMDecoder(vocab_size, embedding_dim=decoder_embedding_dim, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def encode(self, images):
        """Encode image batch to features."""
        encoder_output = self.encoder(images)
        return encoder_output

    def decode_batch(self, encoder_output, token_ids, max_len=300):
        """Decode encoder output to token sequences during training.

        Args:
            encoder_output: (batch, hidden_dim)
            token_ids: (batch, max_len) - ground truth token ids
            max_len: maximum sequence length

        Returns:
            logits: (batch, max_len, vocab_size)
        """
        batch_size = encoder_output.size(0)
        hidden = encoder_output
        cell = torch.zeros_like(encoder_output)

        encoder_output_expanded = encoder_output.unsqueeze(1)  # (batch, 1, hidden_dim)

        logits_list = []

        for t in range(max_len):
            input_token = token_ids[:, t] if t < token_ids.size(1) else torch.zeros(batch_size, dtype=torch.long, device=encoder_output.device)

            logits, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_output_expanded)
            logits_list.append(logits)

        return torch.stack(logits_list, dim=1)

    def generate(self, images, vocab, max_len=300, device='cpu'):
        """Generate token sequences from images.

        Args:
            images: (batch, 1, 640, 480)
            vocab: dict mapping token to index
            max_len: maximum sequence length
            device: device to run on

        Returns:
            generated_tokens: list of token lists
        """
        self.eval()
        with torch.no_grad():
            encoder_output = self.encode(images)
            batch_size = encoder_output.size(0)
            hidden = encoder_output
            cell = torch.zeros_like(encoder_output)

            encoder_output_expanded = encoder_output.unsqueeze(1)

            sos_idx = vocab.get('<SOS>', 1)
            eos_idx = vocab.get('<EOS>', 2)
            unk_idx = vocab.get('<UNK>', 3)
            pad_idx = vocab.get('<PAD>', 0)

            current_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
            generated = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            for t in range(max_len):
                logits, hidden, cell, _ = self.decoder(current_token, hidden, cell, encoder_output_expanded)
                current_token = torch.argmax(logits, dim=1)

                for b in range(batch_size):
                    token_idx = current_token[b].item()
                    if not finished[b]:
                        if token_idx == eos_idx or token_idx == pad_idx:
                            finished[b] = True
                        else:
                            generated[b].append(token_idx)

                if all(finished):
                    break

        return generated


def build_seq2seq_model(vocab_size, device='cpu'):
    """Build and initialize seq2seq model."""
    model = Seq2Seq(
        vocab_size=vocab_size,
        encoder_output_dim=128,
        decoder_embedding_dim=64,
        hidden_dim=128,
    )
    model.to(device)
    return model


def save_model(model, vocab, idx2token, path):
    """Save model and vocabulary."""
    checkpoint = {
        'model_state': model.state_dict(),
        'vocab': vocab,
        'idx2token': idx2token,
    }
    torch.save(checkpoint, path)


def load_model(path, device='cpu'):
    """Load model and vocabulary."""
    checkpoint = torch.load(path, map_location=device)
    vocab = checkpoint['vocab']
    idx2token = checkpoint['idx2token']
    model = build_seq2seq_model(len(vocab), device=device)
    model.load_state_dict(checkpoint['model_state'])
    return model, vocab, idx2token
