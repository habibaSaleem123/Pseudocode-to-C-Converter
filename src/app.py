# src/app.py

import streamlit as st
import torch
import pickle
from model import TransformerSeq2Seq
from vocab import Vocabulary, SPECIAL_TOKENS
from infer import greedy_decode

@st.cache_resource
def load_model_and_vocab():
    # Load vocabularies from pickle files
    with open("src/src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("src/tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)


    # Initialize the model
    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        n_heads=4,
        d_ff=512,
        num_layers=2
    )

    # Choose the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model weights and move model to device
    model.load_state_dict(torch.load("transformer_seq2seq.pt", map_location=device))
    model.to(device)
    model.eval()

    return model, src_vocab, tgt_vocab, device

def main():
    st.title("TranslodeP2C")
    st.write("Enter your pseudocode below:")

    pseudocode_input = st.text_area("Pseudocode", height=200)

    if st.button("Generate Code"):
        if pseudocode_input.strip():
            # Load model, vocab, and device
            model, src_vocab, tgt_vocab, device = load_model_and_vocab()

            # Tokenize the pseudocode
            tokens = pseudocode_input.split()
            src_ids = [src_vocab.word2idx.get(token, src_vocab.word2idx["<unk>"]) for token in tokens]

            # Use the same device that the model is on
            output_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab, device=device)
            output_tokens = [tgt_vocab.idx2word[i] for i in output_ids]
            filtered_tokens = [token for token in output_tokens if token not in SPECIAL_TOKENS]

            st.code(" ".join(filtered_tokens), language='cpp')
        else:
            st.warning("Please enter some pseudocode!")

if __name__ == "__main__":
    main()
