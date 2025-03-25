import argparse
from collections import Counter
import pickle

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.size = 0

    def build_vocab(self, token_lists, min_freq=1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        # Add special tokens first
        for token in SPECIAL_TOKENS:
            self.word2idx[token] = len(self.word2idx)
        # Add remaining tokens if they meet the minimum frequency
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.word2idx:
                self.word2idx[token] = len(self.word2idx)
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        self.size = len(self.word2idx)

    def __len__(self):
        return self.size

    def tokenize_to_ids(self, tokens):
        """Convert a list of tokens to their corresponding indices."""
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

    def ids_to_tokens(self, ids):
        """Convert a list of indices back to tokens."""
        return [self.idx2word.get(i, "<unk>") for i in ids]

def build_and_save_vocab(pairs_file, src_vocab_file, tgt_vocab_file):
    pseudo_tokens_all = []
    code_tokens_all = []
    with open(pairs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                src, tgt = line.split('\t')
            except ValueError:
                print("Line not properly formatted:", line)
                continue
            pseudo_tokens_all.append(src.split())
            code_tokens_all.append(tgt.split())
    src_vocab = Vocabulary()
    src_vocab.build_vocab(pseudo_tokens_all)
    tgt_vocab = Vocabulary()
    tgt_vocab.build_vocab(code_tokens_all)
    print("Pseudo vocab size:", src_vocab.size)
    print("Code vocab size:", tgt_vocab.size)
    with open(src_vocab_file, 'wb') as f:
        pickle.dump(src_vocab, f)
    with open(tgt_vocab_file, 'wb') as f:
        pickle.dump(tgt_vocab, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_file', type=str, required=True, help='Path to the train_pairs.txt file')
    parser.add_argument('--src_vocab_file', type=str, required=True, help='Path to save source vocabulary')
    parser.add_argument('--tgt_vocab_file', type=str, required=True, help='Path to save target vocabulary')
    args = parser.parse_args()
    build_and_save_vocab(args.pairs_file, args.src_vocab_file, args.tgt_vocab_file)

if __name__ == "__main__":
    main()
