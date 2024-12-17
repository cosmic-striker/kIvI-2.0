import torch
import torch.nn as nn
import torch.nn.functional as F

class BARTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_encoder_layers=12, num_decoder_layers=12):
        super(BARTModel, self).__init__()
        
        # Encoder: A stack of Transformer Encoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers
        )
        
        # Decoder: A stack of Transformer Decoder layers
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_decoder_layers
        )
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding (to provide order of tokens in sequence)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1024, d_model))  # Max length of 1024 tokens
        
        # Output layer for generating predictions
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass through the model
        
        Parameters:
        - src: Tensor of shape (batch_size, src_len) [input sequence]
        - tgt: Tensor of shape (batch_size, tgt_len) [target sequence]
        - src_mask: Optional mask for the source sequence
        - tgt_mask: Optional mask for the target sequence
        - src_key_padding_mask: Mask for padding tokens in source sequence
        - tgt_key_padding_mask: Mask for padding tokens in target sequence
        
        Returns:
        - logits: Tensor of shape (batch_size, tgt_len, vocab_size) [predicted tokens]
        """
        
        # Embed the input and target sequences
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        
        # Pass through encoder
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Pass through decoder
        output = self.decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask, 
            memory_mask=src_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Get the logits for each word in the vocabulary
        logits = self.output_layer(output)
        return logits
