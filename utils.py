import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, german, english, device, max_length=50):
    """
    Translate a sentence from German to English using the provided model.

    Args:
        model (nn.Module): The translation model.
        sentence (str or list): The input German sentence as a string or list of tokens.
        german (torchtext.data.Field): German Field object for tokenization.
        english (torchtext.data.Field): English Field object for tokenization.
        device (torch.device): Device to run the model on.
        max_length (int, optional): Maximum length of the output sentence. Defaults to 50.

    Returns:
        list: The translated English sentence as a list of tokens.
    """
    # Load German tokenizer
    spacy_ger = spacy.load("de_core_news_sm")

    # Create tokens using spaCy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in the beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each German token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # Remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    """
    Calculate the BLEU score for the translation model.

    Args:
        data (torchtext.datasets): Dataset to evaluate the model on.
        model (nn.Module): The translation model.
        german (torchtext.data.Field): German Field object for tokenization.
        english (torchtext.data.Field): English Field object for tokenization.
        device (torch.device): Device to run the model on.

    Returns:
        float: The BLEU score.
    """
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # Remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save model checkpoint to file.

    Args:
        state (dict): Dictionary containing model state and optimizer state.
        filename (str, optional): File path to save the checkpoint. Defaults to "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """
    Load model checkpoint from file.

    Args:
        checkpoint (dict): Dictionary containing model state and optimizer state.
        model (nn.Module): The translation model.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
