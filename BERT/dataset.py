import torch
from torch.utils.data import Dataset
from typing import Dict, List, Callable, Tuple

class WikiTextDataset(Dataset):
    
    def __init__(self, data, tok: Callable, sequence_length: int=128):
        """Initialize the WikiTextDataset.

        Args:
            data (List[Dict[str, str]]): The raw data to process.
            tok (Callable): The tokenizer to use.
            sequence_length (int): The maximum sequence length to use.
        """
        self.tokenizer = lambda data, length=sequence_length: tok(data,
                                                                    return_tensors='pt', 
                                                                    max_length=length,
                                                                    truncation=True,
                                                                    padding='max_length',
                                                                    return_token_type_ids=False,
                                                                    is_split_into_words=True)
        
        self.data = []
        for i in range(len(data)):
            self.data.extend(self.splitText(data[i]['text']))
        
        vocab_idxs = torch.ones((self.tokenizer.vocab_size,))
        #zero out sepcial tokens
        vocab_idxs[0:1996] = 0
        #length of valid, litteral tokens
        v_len = len(vocab_idxs[1996:])
        #equal probability distribution over tokens that are litterals.
        self.sample_from_vocab = (vocab_idxs / v_len)


    def __len__(self) -> int:
        """Returns the length of the list data

        Returns:
            int: List length
        """
        return self.data.__len__()
    
    @staticmethod
    def splitText(text: str, seq_len: int=128) -> List[List[str]]:
        """Split the text into sequences of a given length.
        Args:
            text (str): String to be splitted into sequences.
            seq_len (int): Length of splitted sequence. Default 128.
        Returns:
            List[List[str]]: List containing string splitted into sublists of length seq_len.
        """
        text_split = text.split()
        text_split_len = len(text_split)
        
        return [text_split[i:i+seq_len] for i in range(0, text_split_len, seq_len)]
    
    def encode(self, data: List[str]) -> Dict[torch.Tensor, torch.Tensor]:
        """Encode the input data.

        Args:
            data (List[str]): The data to encode.

        Returns:
            A dictionary containing tensors for the encoded tokens, the attention mask, and the original tokens.
        """
        tokenized = self.tokenizer(data)
        #encoded tokens
        tokens = tokenized['input_ids'].reshape(-1)
        #mask
        att_mask = tokenized['attention_mask'].reshape(-1)
        #filter [CLS] [SEP] tokens to not replace them with [MASK]
        special_tokens_filter = tokens.eq(102) | tokens.eq(101)
        #equal probability distribution over tokens excluding [SEP], [CLS]
        samples = att_mask.masked_fill(special_tokens_filter, 0) * 0.15
        #select index with probability 15%
        selected = torch.bernoulli(samples)
        #create tensor wit numbers in range 0-1, this is representing probabilities
        values = torch.rand(selected.shape)
        #auxilary mask for selecting probabilities from values
        m = selected == 1
        #if number is smaller than 0.8 replace token with [mask], if between 0.8,0.9 replace with random token, else do not replace
        selected[m] = torch.where(values[m] < 0.8, 2., torch.where(values[m] < 0.9, 3., 1.))
        #apply masks to tokens
        x = tokens.masked_fill(selected == 2, 103)
        x = x.masked_fill(selected == 3, torch.multinomial(self.sample_from_vocab, num_samples=1).item())
        #update mask returned from tokenizer
        mask = att_mask.masked_fill(selected != 0, 0)
        
        return x.unsqueeze(0), mask.unsqueeze(0), tokens.unsqueeze(0)
    
    def __getitem__(self, idx: int) -> Dict[torch.Tensor, torch.Tensor]:
        
        return self.encode(self.data[idx])