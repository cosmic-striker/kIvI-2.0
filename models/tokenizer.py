import os
import sentencepiece as spm

class Tokenizer:
    def __init__(self, tokenizer_path=None, vocab_size=30522, train_data_path=None):
        """
        Initialize Tokenizer.
        :param tokenizer_path: Path to a pre-trained SentencePiece model.
        :param vocab_size: Vocabulary size for training.
        :param train_data_path: Path to data for training a new tokenizer.
        """
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.train_data_path = train_data_path

        self.sp = spm.SentencePieceProcessor()

        # Load or train the tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}...")
            self.sp.Load(tokenizer_path)
        elif train_data_path:
            print(f"Training tokenizer with data from {train_data_path}...")
            self.train_tokenizer()
        else:
            raise ValueError("Either tokenizer_path or train_data_path must be provided.")

    def train_tokenizer(self):
        """
        Train a SentencePiece tokenizer on the training data.
        """
        model_prefix = "tokenizer_model"
        spm.SentencePieceTrainer.train(
            input=self.train_data_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,  # Cover almost all characters
            model_type='bpe'  # BPE (Byte Pair Encoding) for subword tokenization
        )
        self.tokenizer_path = f"{model_prefix}.model"
        print(f"Tokenizer trained and saved at {self.tokenizer_path}.")
        self.sp.Load(self.tokenizer_path)

    def encode(self, text):
        """
        Encode text into token IDs.
        :param text: Input string.
        :return: List of token IDs.
        """
        return self.sp.EncodeAsIds(text)

    def decode(self, ids):
        """
        Decode token IDs back into a string.
        :param ids: List of token IDs.
        :return: Decoded string.
        """
        return self.sp.DecodeIds(ids)

    def save_tokenizer(self, output_path):
        """
        Save the trained tokenizer model to a file.
        :param output_path: Path to save the tokenizer model.
        """
        if self.tokenizer_path and os.path.exists(self.tokenizer_path):
            os.rename(self.tokenizer_path, output_path)
            print(f"Tokenizer saved to {output_path}.")
        else:
            raise FileNotFoundError("Tokenizer model not found for saving.")
