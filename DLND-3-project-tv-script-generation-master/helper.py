import os
import pickle
import torch


SPECIAL_WORDS = {'PADDING': '<PAD>'}


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)

def median(array):
    if len(array) < 1:
        return(None)
    if len(array) % 2 == 0:
        median = (array[len(array)//2-1: len(array)//2+1])
        return sum(median) / len(median)
    else:
        return(array[len(array)//2])
    
def mode(array):
    '''
    returns a set containing valid modes
    returns a message if no valid mode exists
      - when all numbers occur the same number of times
      - when only one number occurs in the list 
      - when no number occurs in the list 
    '''
    most = max(map(array.count, array)) if array else None
    mset = set(filter(lambda x: array.count(x) == most, array))
    return mset if set(array) - mset else "list does not have a mode!" 
