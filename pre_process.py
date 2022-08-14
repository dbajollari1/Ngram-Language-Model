import string
import itertools

def prep_data(filename):

    # Load file
    with open(filename, encoding="utf8") as f:
        lines = [line.rstrip() for line in f]
    # print("No of sentences in Corpus: "+ str(len(lines)))

    # Tokenize data
    lines = [i.strip("''").split(" ") for i in lines] 

    for i in range(len(lines)):
        # lines[i] = [''.join(c for c in s if c not in string.punctuation) for s in lines[i]] # remove punctuations
        lines[i] = [s for s in lines[i] if s]          # removes empty strings
        lines[i] = [word.lower() for word in lines[i]] # lower case
        lines[i] += ['</s>']                           # Append </s> at the end of each sentence in the corpus
        #lines[i].insert(0, '<s>')                      # Append <s> at the beginning of each sentence in the corpus

    return lines

def vocabulary(dataset):
    dataset_vocab = set(itertools.chain.from_iterable(dataset))
    dataset_vocab.remove('</s>')
    dataset_vocab = list(dataset_vocab)
    dataset_vocab.append('</s>')
    return dataset_vocab

def freq_of_unique_words(lines):
    bag_of_words = list(itertools.chain.from_iterable(lines)) # change the nested list to one single list
    word_count = {} # dictionary of word:count
    for word in bag_of_words:
        if word in word_count :
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count, len(bag_of_words)

