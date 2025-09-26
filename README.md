# Shakespeare Text Generator

An LSTM-based Recurrent Neural Network implementation for generating Shakespearean-style text using TensorFlow and Keras.

## 📖 Overview

This project implements a character-level text generation model trained on Shakespeare's complete works to create AI-generated text that imitates the Bard's distinctive writing style. The model uses Long Short-Term Memory (LSTM) networks to learn patterns in Shakespeare's language and generate coherent, contextually appropriate text.

## 🎯 Project Objectives

- **Text Generation**: Create an AI model capable of generating Shakespeare-style text
- **Pattern Recognition**: Learn and replicate the linguistic patterns and structures in Shakespeare's works
- **Deep Learning Implementation**: Demonstrate practical application of LSTM networks for sequential data
- **Language Modeling**: Understand character-level language modeling techniques

## 📊 Dataset

- **Source**: Complete works of William Shakespeare
- **Size**: 5+ million characters
- **Format**: Plain text file (`shakespeare.txt`)
- **Vocabulary**: 84 unique characters including letters, punctuation, and special characters

## 🏗️ Model Architecture

### Why LSTM?

The project uses LSTM (Long Short-Term Memory) networks instead of traditional RNNs to address the **vanishing gradient problem**:

- **Vanishing Gradient Issue**: In standard RNNs with multiple layers, gradient values decrease exponentially during backpropagation, causing the partial derivatives to approach zero
- **LSTM Solution**: LSTM cells use gates (forget, input, output) to control information flow, maintaining long-term dependencies effectively
- **Trade-off Consideration**: While GRU (Gated Recurrent Unit) uses less memory and is faster, LSTM provides better accuracy for longer sequences, making it ideal for text generation

### Network Structure

```
Model Architecture:
├── Embedding Layer (84 dimensions)
├── LSTM Layer 1 (1026 units, return_sequences=True, dropout=0.4)
├── LSTM Layer 2 (500 units, return_sequences=True, dropout=0.4)
└── Dense Output Layer (84 units - vocabulary size)

Total Parameters: 7,662,684
```

### Key Components

1. **Embedding Layer**: Maps character indices to dense vectors (84-dimensional)
2. **Stacked LSTM Layers**: Two LSTM layers for hierarchical feature learning
3. **Dropout Regularization**: 40% dropout rate to prevent overfitting
4. **Dense Output Layer**: Maps LSTM output to vocabulary probabilities

## 🔧 Technical Implementation

### Text Preprocessing Pipeline

1. **Character Mapping**: Create bidirectional mappings between characters and indices
2. **Sequence Creation**: Generate input-target pairs with 250-character sequences
3. **Vectorization**: Convert text to numerical representations
4. **Batch Generation**: Create batches of 128 sequences for efficient training

### Critical Hyperparameters

- **Sequence Length**: 250 characters
- **Batch Size**: 128 (critical for pattern recognition across samples)
- **Embedding Dimension**: 84
- **LSTM Units**: 1026 (first layer), 500 (second layer)
- **Buffer Size**: 8000 (for dataset shuffling)
- **Training Epochs**: 40

### Batch Size Importance

Batch size is crucial for LSTM and CNN models because:
- **Pattern Recognition**: Models need multiple samples to identify common patterns and features
- **Gradient Stability**: Larger batches provide more stable gradient estimates
- **Feature Learning**: Batch processing enables the model to learn generalizable features across different text segments

## 📈 Training Results

- **Final Loss**: 1.07 (after 40 epochs)
- **Training Parameters**: 7M+ parameters optimized
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Training Time**: ~73 minutes (40 epochs)

### Training Progress
```
Epoch 1/40 - Loss: 2.9642
Epoch 10/40 - Loss: 1.2420
Epoch 20/40 - Loss: 1.1109
Epoch 30/40 - Loss: 1.0520
Epoch 40/40 - Loss: 1.0143
```

## 🚀 Usage

### Prerequisites
```bash
pip install tensorflow numpy pandas
```

### Running the Model

1. **Load the trained model**:
```python
from tensorflow.keras.models import load_model
model = load_model('shakespeare_gen1.h5')
```

2. **Generate text**:
```python
generated_text = generate_text(model, "JULIET ", gen_size=800)
print(generated_text)
```

### Text Generation Parameters

- **start_seed**: Initial text prompt (string)
- **gen_size**: Number of characters to generate (default: 100)
- **temperature**: Controls randomness (1.0 = balanced, <1.0 = conservative, >1.0 = creative)

## 📝 Example Output

**Input**: "JULIET "

**Generated Output**:
```
JULIET AND             Exit. Soldiers.

          Enter PROTEUS, VALENTINE, and SHYLOCK

               EO-enter CHARMIAN, IACHIO, AUMERLE, CHILD and ATTENDANTS

  CLARENCE. O, let me sing your Grace!
    What, art thou to our conscience?
  MENELAUS. If I can rush so well,
    Impromish your equisore.
  LEONTES. Come, come, pardon; let 't it down.
...
```

## 🎭 Model Performance Analysis

The model successfully captures:
- **Character Names**: Generates appropriate Shakespearean character names
- **Dialog Structure**: Maintains proper formatting for plays (character names, stage directions)
- **Language Style**: Produces text with Elizabethan English patterns
- **Dramatic Elements**: Includes stage directions and scene transitions
- **Poetic Rhythm**: Maintains some semblance of iambic pentameter in places

## 📂 Project Structure

```
Shakespeare-Text-Generator/
├── README.md
├── Shakespeare_text_generator.ipynb    # Main implementation notebook
├── shakespeare.txt                     # Training dataset
└── shakespeare_gen1.h5                # Trained model weights
```

## 🔬 Technical Deep Dive

### Loss Function
**Sparse Categorical Crossentropy** is used because:
- Target labels are integers (character indices) rather than one-hot encoded vectors
- More memory efficient for large vocabularies
- `from_logits=True` since the model outputs raw logits

### Temperature Sampling
The generation function uses temperature-based sampling:
- **Low Temperature (0.5)**: More predictable, conservative text
- **High Temperature (1.5)**: More creative, potentially incoherent text
- **Balanced (1.0)**: Good balance between creativity and coherence

### Stateful LSTM
The model uses `stateful=True` to:
- Maintain hidden states across batches during training
- Enable continuous sequence modeling
- Improve learning of long-term dependencies

## 🛠️ Future Enhancements

- [ ] Implement word-level generation for better semantic coherence
- [ ] Add attention mechanisms for improved context awareness
- [ ] Experiment with transformer architectures
- [ ] Fine-tune hyperparameters for specific Shakespearean works
- [ ] Implement beam search for better text generation
- [ ] Add genre-specific generation (sonnets, plays, etc.)

## 📚 Key Learnings

1. **LSTM Advantages**: Superior performance over RNNs for long sequences
2. **Batch Processing**: Critical for effective pattern learning in neural networks
3. **Character-Level Modeling**: Enables creative text generation with proper formatting
4. **Regularization**: Dropout is essential for preventing overfitting in large models
5. **Temperature Control**: Allows fine-tuning between creativity and coherence

## 🔗 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Python**: Programming language
- **Jupyter Notebook**: Development environment

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

---

*"All the world's a stage, and all the men and women merely players." - William Shakespeare*

**Note**: This project is for educational purposes and demonstrates the application of LSTM networks in natural language processing and text generation.