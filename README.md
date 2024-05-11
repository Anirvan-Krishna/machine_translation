# Neural Machine Translation with Transformers

This repository contains Python code for a Neural Machine Translation (NMT) system implemented using the Transformer architecture. The system translates text from German to English using a transformer model.

## Overview

The main functionality of the code is to translate German sentences to English using a Transformer-based neural network model. The translation model is implemented using PyTorch, a popular deep learning framework. The repository also includes a `main.py` script for running the translation process, along with a pre-trained model saved as a pickle file.

## Dependencies

To run the code in this repository, you'll need the following libraries:

- Python (>=3.6)
- PyTorch
- TorchText(0.6.0)
- spaCy
- tqdm

You can install the required dependencies using pip:
```
pip install torch 
pip install torchtext==0.6.0 
pip install spacy tqdm
```

Additionally, you'll need to download the German and English spaCy models:

```
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Model Structure
The translation model is based on the Transformer architecture, which is a state-of-the-art model for sequence-to-sequence tasks. Here's an overview of the model structure:

Embedding Layers: Convert input tokens into dense vectors.
Transformer Encoder: Process the input sequence and generate context representations.
Transformer Decoder: Generate output sequence based on the context representations from the encoder.
Linear Layer: Map the decoder outputs to the target vocabulary space.

## Training Details
The model was trained using the Multi30k dataset, which contains parallel text data in English and German. Here are some key training details:

- Optimizer: Adam optimizer with a learning rate of 3e-4.
- Loss Function: Cross-Entropy Loss with ignore index for padding tokens.

### Training Hyperparameters:
- Number of epochs: 100
- Batch size: 32
- Model Hyperparameters:
- Embedding size: 512
- Number of attention heads: 8
- Number of encoder and decoder layers: 3
- Dropout probability: 0.10
- Maximum sequence length: 100

### Loss plot
![Loss Plot](/images/loss_plot.png)

The model is trained for a total of 100 epochs. The BLEU score achieved after 100 epoch training is 32.12 and it is estimated to increase with further training


## Usage
1. Clone this repository to your local machine:
```
git clone <repository_url>
```
2. Install the required dependencies as mentioned above.
3. Run the main.py script to translate German sentences to English:
```
python main.py
 ```


## Files

- `main.py`: Main Python script for translating German sentences to English using the pre-trained Transformer model.
- `model.pkl`: Pickle file containing the pre-trained Transformer model.
- `utils.py`: Consists of utility files for saving checkpoints, loading checkpoints and evaluating BLEU score
- `README.md`: This README file providing an overview of the repository.

