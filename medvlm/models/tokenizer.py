from transformers import AutoTokenizer
import torch 

class Tokenizer:
    def __init__(self, model_name="microsoft/BiomedVLP-CXR-BERT-specialized", max_length=512, trust_remote_code=True):
        """Wrapper to ensure compatibility """
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.bos_token_id = tokenizer.cls_token_id  # Use <CLS> as <BOS> 
        self.eos_token_id = tokenizer.sep_token_id  # Use <SEP> as <EOS>
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer

    def __call__(self, text):
        """
        Tokenizes the input text, applies padding and truncation, and returns input IDs
        with <BOS> and optional <EOS> tokens included.
        
        :param text: Input text to tokenize.
        :return: Tokenized input IDs with <BOS> and optional <EOS>.
        """
        # Tokenize the text without adding special tokens
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,  # Avoid automatic addition of special tokens
            return_tensors='pt',
            max_length=self.max_length - 1,  # Reserve space for <BOS> and optionally <EOS>
            padding=False,
            truncation=True
        ).input_ids[0]
        
        # Initialize with <BOS> token
        bos_tokens = [self.bos_token_id] + tokens.tolist()
        
        # Add <EOS> only if it fits within max_length
        if len(bos_tokens) < self.max_length:
            bos_tokens.append(self.eos_token_id)
        
        # Pad the sequence if needed
        padded_tokens = bos_tokens[:self.max_length]  # Truncate if exceeding max_length
        padding_length = max(0, self.max_length - len(padded_tokens))
        padded_tokens.extend([self.tokenizer.pad_token_id] * padding_length)  # Add padding tokens

        return torch.tensor(padded_tokens)