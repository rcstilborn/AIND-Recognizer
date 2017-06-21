import warnings
from asl_data import SinglesData
import logging
import math

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.DEBUG)
    probabilities = []
    guesses = []
    # for each unknown word in the test set score it with each model provided.
    # Pick the one with the best fit
    
    for item in range(0,test_set.num_items):
        logging.debug("Recognizing sample {} with these sequences {}".format(item,test_set.get_item_Xlengths(item)))
        probs = dict()
        X, lengths = test_set.get_item_Xlengths(item)
        for word,model in models.items():
            logging.debug("  Comparing to {}".format(word))
            try:
                score = model.score(X, lengths)
                logging.debug("    Got this score {}:{}".format(word,score))
                probs[word]=score
            except Exception as e:
                logging.warning("{} caught while scoring model for word {}: {}".format(type(e),word,e))
                probs[word] = -math.inf
                pass
        probabilities.append(probs)
        if len(probs)==0:
            guesses.append("None")
            logging.debug("  No results found for item {}!".format(item))
        else:
            best_guess=max(probs, key=lambda key: probs[key])
            guesses.append(best_guess)
            logging.debug("  Best option {}:{}".format(best_guess, probs[best_guess]))
            
    return probabilities, guesses


def get_WER(guesses: list, test_set: SinglesData):
    """ Calculates WER
    
    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return: WER

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    return float(S) / float(N)
