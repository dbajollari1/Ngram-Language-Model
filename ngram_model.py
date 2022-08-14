import itertools
import math
class NgramModel:

    def __init__(self, dataset, vocab_size, n, laplace=1):
        self.n = n
        self.dataset = dataset 
        self.laplace = laplace  
        self.vocab_size = vocab_size
        if n == 1:
            self.total_words = len(list(itertools.chain.from_iterable(dataset))) # use only for monogram
        else:
            self.prev_ngram_frequencies = self.generate_ngram_frequencies(n-1) 
            self.unique_tokens = list(set(self.prev_ngram_frequencies.keys()))
        self.ngram_frequencies = self.generate_ngram_frequencies(n)
        self.model  = self.buildModel()


    def buildModel(self):
        # For each n-gram, calculate its conditional probability in the training text
        ngram_probabilities = dict() 

        # apply laplace smoothing when requested (self.laplace = 1)
        vocab_size = 0
        k = 0
        if self.laplace == 1:
            k = 1
            vocab_size = self.vocab_size

        for token, token_count in self.ngram_frequencies.items():
            if self.n == 1: # monogram
                ngram_probabilities[token] = token_count / self.total_words
            else:
                nominator_count = token_count # self.ngram_frequencies[token] if token in self.ngram_frequencies else 0
                prev_token = token[:self.n-1]
                denominator_count = self.prev_ngram_frequencies[prev_token] if prev_token in self.prev_ngram_frequencies else 0
                if denominator_count > 0 or vocab_size > 0:
                    ngram_probabilities[token] = (nominator_count + k)/(denominator_count + (k * vocab_size))

        return ngram_probabilities


    def generate_ngram_frequencies(self, n):
        ngram_frequencies = dict()

        for sentence in self.dataset:           
            # for i in range(len(sentence)):
            #     key = tuple(sentence[i:i+n])
            #     if len(key) == n:
            #         if key in ngram_frequencies:
            #             ngram_frequencies[key] += 1
            #         else:
            #             ngram_frequencies[key] = 1
            for key in self._generate_ngram(sentence, n):
                if key in ngram_frequencies:
                    ngram_frequencies[key] += 1
                else:
                    ngram_frequencies[key] = 1
        return ngram_frequencies


    def _generate_ngram(self, sentence, n):
        ngram= []

        if len(sentence) < n: # sentcence is shorter than number of n-grams
            key = tuple(sentence)
            ngram.append(key)
            return ngram

        for i in range(len(sentence)):
            key = tuple(sentence[i:i+n])  
            if len(key) == n:
                ngram.append(key)
        return ngram


    def perplexity(self, test_data):
        # Calculate the perplexity of the model against a given test corpus.
        N = len(test_data)
        sentencePPLs = []
        for sentence in test_data:
            probabilities = []
            test_token = self._generate_ngram(sentence, self.n)
            ngramLen = len(test_token)

            for token in test_token:
                if token not in self.model:
                    # probabilities.append(0)
                    continue
                else:
                    probabilities.append(self.model[token])
            probs = math.prod(probabilities)
            if probs == 0:
                sentencePPLs.append(0)
            else:
                sentencePPLs.append(probs**(-1/ngramLen))

        ppl = sum(sentencePPLs)/N 
        #print("ngram-" + str(self.n) + ": " + str(ppl))        
        return ppl

    def prodict_next_word(self, token):
        word = ""
        high_prob=0
        size = len(token)
        for key, prob in self.model.items():
            # if size == 0: # first iteration, no word yet - pick word with highest probability
            #     if prob > high_prob:
            #         word = key[0]
            #         high_prob = prob 
            if key[0:size] == token:
                if prob > high_prob:
                    word = key[size]
                    high_prob = prob                
        return word


    def generate_text(self, word_count, seed):
        n = self.n
        ngram_token_queue = []
        ngram_token_queue.append(seed)
        result = []
        result.append(seed)
        for _ in range(word_count):
            word = self.prodict_next_word(tuple(ngram_token_queue))
            if word == "</s>": # end of sentence
                ngram_token_queue = []
                ngram_token_queue.append(seed)
                result.append("\n") # add new line
            else:
                result.append(word)
                ngram_token_queue.append(word)
            if len(ngram_token_queue) == self.n:
                ngram_token_queue.pop(0) # remove first from queue
                 
        return ' '.join(result)

