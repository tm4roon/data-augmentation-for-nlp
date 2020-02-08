import torch
from collections import OrderedDict


class PreDefinedVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    SPECIAL_TOKEN_ATTRIBUTES = [
        'unk_token', 'pad_token', 'bos_token', 'eos_token',
        'mask_token', 'sep_token', 'cls_token',
    ]


    def __init__(self, vocab_file, **kwargs):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            vocab_file: A path to vocabulary file.
            unk_token: The string token used to represent OOV words.
            pad_token: The string token used as padding. 
            bos_token: A token that will be prepended to every example using this field
            eos_token: A token that will be appended to every example using this field
            mask_token: The string token used as masking.
            sep_token: The string token used to separate sentences.
            cls_token: The string token used to classification.
        """

        with open(vocab_file, 'r') as f:
            self.itos = [w.rstrip() for w in f]

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKEN_ATTRIBUTES and value is not None:
                setattr(self, key, value)
                setattr(self, key.split('_')[0] + '_id', self.itos.index(value))
        self.stoi = OrderedDict({w: i for i, w in enumerate(self.itos)})


    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(self.unk_token))

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class WordpieceTokenizer(object):
    """WordPiece tokenization."""

    def __init__(self, vocab, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = vocab.unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab.stoi:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def string2ids(self, tokens):
        return [self.vocab(w) for w in tokens]

    def ids2string(self, arr):
        return [self.vocab.itos[i] for i in ids]

    def encode(self, x):
        tokenized = self.tokenize(x)
        return string2ids(tokenized)

    def decode(self, arr):
        if hasattr(self.vocab, 'eos_id') and tokens in self.vocab.eos_id:
            tokens = tokens[:tokens.index(self.vocab.eos_id)]
        tokens = self.ids2string(arr)
        return  reduce(lambda x, y: f"{x}{y.lstrip('##')}"  
                       if y.startswith('##') else f"{x} {y}", tokens)

