import re

import torch
import torch.nn as nn
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence


VOCAB = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<sil>",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    " ", ".", ",", "!", "?", "'", ";", "-", ":"
]


class CharEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.vocab = VOCAB
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self._dummy = nn.Parameter(torch.zeros(1))
        self.saveable = False
        
    def forward(self, text: list[str]) -> LongTensor:
        r"""Convert text into IDs.

        b: batch_size
        l: seq_len
        d: dim

        Args:
            text: list[str]

        Returns:
            ids: (b, l, d)
        """
        device = next(self.parameters()).device
        text = [self.normalize_text(t) for t in text]

        ids = [LongTensor([self.get_id(c) for c in t]) for t in text]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.char2id["<pad>"]).to(device)
        mask = ids != self.char2id["<pad>"]
        return ids, mask


    def get_id(self, char):
        if char in self.char2id:
            id = self.char2id[char]
        else:
            id = self.char2id["<unk>"]
            print(f"{char} is not in vocab.")

        return id


    def normalize_text(self, text: str) -> str:
        return "".join(c for c in text.lower() if c in self.vocab)