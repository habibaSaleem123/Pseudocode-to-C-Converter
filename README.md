# Pseudo2Cpp: AI-Powered Pseudocode to C++ Converter  

🚀 **Pseudo2Cpp** is an AI-driven system that translates **pseudocode** into structured **C++ programs** using a **Transformer-based Seq2Seq model**. Designed for developers, students, and coding enthusiasts, it automates the conversion process, bridging the gap between algorithm design and implementation.  

## 🔥 Features  
✅ **AI-Powered Code Generation** – Transforms pseudocode into optimized C++ code using deep learning.  
✅ **Transformer-Based Architecture** – Leverages a custom-built seq2seq Transformer for accurate conversions.  
✅ **Smart Vocabulary Management** – Tokenizes and structures data for efficient model training.  
✅ **Customizable Training Pipeline** – Fine-tune hyperparameters to optimize performance.  
✅ **Greedy Decoding for Inference** – Ensures fast and accurate C++ code generation.  
✅ **Interactive Streamlit UI** – User-friendly web interface for seamless code conversion.  

## 🛠 Installation  

### **Prerequisites**  
Ensure you have the following installed:  
- Python **3.8+**  
- PyTorch  
- Streamlit  
- tqdm  

### **Setup**  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/absarraashid3/pseudo2cpp.git  
cd pseudo2cpp  
pip install -r requirements.txt  
```  

## 🚀 Usage  

### **Step 1: Preprocess Data**  
Convert your TSV training data into paired pseudocode-C++ format:  
```bash
python src/preprocess.py --input_tsv "data/train/split/spoc-train-train.tsv" --output_txt "data/train_pairs.txt"
```  

### **Step 2: Build Vocabulary**  
Generate vocabulary files for training:  
```bash
python src/vocab.py --pairs_file "data/train_pairs.txt" --src_vocab_file "src/src_vocab.pkl" --tgt_vocab_file "src/tgt_vocab.pkl"
```  

### **Step 3: Train the Model**  
Train the Transformer model for pseudocode-to-C++ conversion:  
```bash
python src/train.py --pairs_file "data/train_pairs.txt" --src_vocab_file "src/src_vocab.pkl" --tgt_vocab_file "src/tgt_vocab.pkl" --epochs 10 --batch_size 8
```  

### **Step 4: Convert Pseudocode to C++**  
Generate C++ code from pseudocode input:  
```bash
python src/infer.py --model_checkpoint transformer_seq2seq.pt --src_vocab_file "src/src_vocab.pkl" --tgt_vocab_file "src/tgt_vocab.pkl" --pseudocode "read n print factorial of n"
```  

## 🎨 Web Application  
Launch the **interactive Streamlit UI** to test real-time conversions:  
```bash
streamlit run src/app.py
```  

## 🎯 Why Use Pseudo2Cpp?  
💡 **Automates Code Writing** – Eliminates manual C++ translation.  
⚡ **Accelerates Development** – Ideal for students, researchers, and professionals.  
📈 **Enhances Learning** – Understand C++ syntax through AI-generated code.  

🔥 **Turn your ideas into code effortlessly!**  
