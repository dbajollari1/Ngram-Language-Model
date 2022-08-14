import math
from pre_process import *
from ngram_model import *
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

def build_uniform_unigram_model(unigram_frequencies,vocab_size):
    unigram_probabilities = dict() 

    for token in unigram_frequencies:
        unigram_probabilities[token] = 1 / vocab_size

    return unigram_probabilities

def build_relative_unigram_model(unigram_frequencies,total_tokens):
    unigram_probabilities = dict() 

    for token,token_count in unigram_frequencies.items():
        unigram_probabilities[token] = token_count / total_tokens

    return unigram_probabilities

def perplexity(model, test_data):
    # Calculate the perplexity of the model against a given test corpus.
    N = len(test_data)

    sentencePPLs = []
    for sentence in test_data:
        probabilities = []
        sentenceLen = len(sentence)
        for unigram in sentence:
            if unigram not in model:
                continue
            else:
                probabilities.append(model[unigram])
        probs = math.prod(probabilities)
        if probs == 0:
            sentencePPLs.append(0)
        else:
            sentencePPLs.append(probs**(-1/sentenceLen))
        
    return sum(sentencePPLs)/N

def write_document(file_name, generated_text):
    with open(file_name, "w", encoding="utf8") as f:
        f.write(generated_text)

def main():

    # load in datasets
    dataset_ted = prep_data("./datasets/train/ted.txt")
    dataset_vocab_ted = vocabulary(dataset_ted)
    word_frequency_ted, total_words_ted = freq_of_unique_words(dataset_ted)
    dataset_reddit = prep_data("./datasets/train/reddit.txt")
    dataset_vocab_reddit = vocabulary(dataset_reddit)
    word_frequency_reddit, total_words_reddit = freq_of_unique_words(dataset_reddit)
    
    # build models for part 1 and 2 
    uniform_unigram_model = build_uniform_unigram_model(word_frequency_ted, len(dataset_vocab_ted))
    relative_unigram_model = build_relative_unigram_model(word_frequency_ted, total_words_ted)

    # load test files
    test_news_dataset = prep_data("./datasets/test/test.news.txt")
    test_reddit_dataset = prep_data("./datasets/test/test.reddit.txt")
    test_ted_dataset = prep_data("./datasets/test/test.ted.txt")


    # calculate PPL for part 1 and 2 
    testRedditUniformModelPPL = perplexity(uniform_unigram_model, test_reddit_dataset)
    print("Uniform Unigram Model PPL using test.reddit.txt: " + str(round(testRedditUniformModelPPL,4)))
    testTedUniformModelPPL = perplexity(uniform_unigram_model, test_ted_dataset)
    print("Uniform Unigram Model PPL using test.ted.txt: " + str(round(testTedUniformModelPPL,4)))
    testNewsUniformModelPPL = perplexity(uniform_unigram_model, test_news_dataset)
    print("Uniform Unigram Model PPL using test.news.txt: " + str(round(testNewsUniformModelPPL,4)))

    testRedditRelativeModelPPL = perplexity(relative_unigram_model, test_reddit_dataset)
    print("Relative Unigram Model PPL using test.reddit.txt: " + str(round(testRedditRelativeModelPPL,4)))
    testTedRelativeModelPPL = perplexity(relative_unigram_model, test_ted_dataset)
    print("Relative Unigram Model PPL using test.ted.txt: " + str(round(testTedRelativeModelPPL,4)))
    testNewsRelativeModelPPL = perplexity(relative_unigram_model, test_news_dataset)
    print("Relative Unigram Model PPL using test.news.txt: " + str(round(testNewsRelativeModelPPL,4)))

    print("\n***Part 1 and 2 Done***\n")

    # Part 3 plot - Create two (ted.txt and reddit.txt) language models using Use values of n from 1 to 7.
    #               and apply to each test file (ted, reddit, news)
    ngram_ted_PPL_ted = []
    ngram_ted_PPL_reddit= []
    ngram_ted_PPL_news = []
    ngram_reddit_PPL_ted = []
    ngram_reddit_PPL_reddit = []
    ngram_reddit_PPL_news = []
    
    for i in range(1,8):
        # Ted Model
        ngramModelTed = NgramModel(dataset_ted, len(dataset_vocab_ted), i, 0)
        ngram_ted_PPL_ted.append(ngramModelTed.perplexity(test_ted_dataset))
        ngram_ted_PPL_reddit.append(ngramModelTed.perplexity(test_reddit_dataset))
        ngram_ted_PPL_news.append(ngramModelTed.perplexity(test_news_dataset))

        # Reddit Model
        ngramModelReddit = NgramModel(dataset_reddit, len(dataset_vocab_reddit), i, 0)
        ngram_reddit_PPL_ted.append(ngramModelReddit.perplexity(test_ted_dataset))
        ngram_reddit_PPL_reddit.append(ngramModelReddit.perplexity(test_reddit_dataset))
        ngram_reddit_PPL_news.append(ngramModelReddit.perplexity(test_news_dataset))
    
    x = [1,2,3,4,5,6,7]
    plt1.plot(x, ngram_ted_PPL_ted, label = "Ted Ngram Model PPL on test.ted.txt", linestyle="-")
    plt1.plot(x, ngram_ted_PPL_reddit, label = "Ted Ngram Model PPL on test.reddit.txt", linestyle="-")
    plt1.plot(x, ngram_ted_PPL_news, label = "Ted Ngram Model PPL on test.news.txt", linestyle="-")
    plt1.plot(x, ngram_reddit_PPL_ted, label = "Reddit Ngram Model PPL on test.ted.txt", linestyle="-")
    plt1.plot(x, ngram_reddit_PPL_reddit, label = "Reddit Ngram Model PPL on test.reddit.txt", linestyle="-")
    plt1.plot(x, ngram_reddit_PPL_news, label = "Reddit Ngram Model PPL on test.news.txt", linestyle="-")

    plt1.legend()
    plt1.title('Plot for Part 3')
    plt1.show()

    print("\n***Part 3 Done***\n")

    # Part 4 plot - same as part 3 but apply Laplace smoothing
    ngram_ted_PPL_ted = []
    ngram_ted_PPL_reddit= []
    ngram_ted_PPL_news = []
    ngram_reddit_PPL_ted = []
    ngram_reddit_PPL_reddit = []
    ngram_reddit_PPL_news = []
    
    for i in range(2,8):
        # Ted Model
        ngramModelTed = NgramModel(dataset_ted, len(dataset_vocab_ted), i, 1)
        ngram_ted_PPL_ted.append(ngramModelTed.perplexity(test_ted_dataset))
        ngram_ted_PPL_reddit.append(ngramModelTed.perplexity(test_reddit_dataset))
        ngram_ted_PPL_news.append(ngramModelTed.perplexity(test_news_dataset))

        # Reddit Model
        ngramModelReddit = NgramModel(dataset_reddit, len(dataset_vocab_reddit), i, 1)
        ngram_reddit_PPL_ted.append(ngramModelReddit.perplexity(test_ted_dataset))
        ngram_reddit_PPL_reddit.append(ngramModelReddit.perplexity(test_reddit_dataset))
        ngram_reddit_PPL_news.append(ngramModelReddit.perplexity(test_news_dataset))
    
    x = [2,3,4,5,6,7]
    plt2.plot(x, ngram_ted_PPL_ted, label = "Ted Ngram Model PPL on test.ted.txt", linestyle="-")
    plt2.plot(x, ngram_ted_PPL_reddit, label = "Ted Ngram Model PPL on test.reddit.txt", linestyle="-")
    plt2.plot(x, ngram_ted_PPL_news, label = "Ted Ngram Model PPL on test.news.txt", linestyle="-")
    plt2.plot(x, ngram_reddit_PPL_ted, label = "Reddit Ngram Model PPL on test.ted.txt", linestyle="-")
    plt2.plot(x, ngram_reddit_PPL_reddit, label = "Reddit Ngram Model PPL on test.reddit.txt", linestyle="-")
    plt2.plot(x, ngram_reddit_PPL_news, label = "Reddit Ngram Model PPL on test.news.txt", linestyle="-")

    plt2.legend()
    plt2.title('Plot for Part 4 (with laplace smoothing)')
    plt2.show()

    print("\n***Part 4 Done***\n")


    # Generate text - using ngram size 7 with no laplace smoothing on ted and reddit datasets

    ngramModelTed = NgramModel(dataset_ted, len(dataset_vocab_ted), 7, 0) # 
    ngramModelReddit = NgramModel(dataset_reddit, len(dataset_vocab_reddit), 7, 0)

    unigramModelTed = NgramModel(dataset_ted, len(dataset_vocab_ted), 1, 0)
    seed = max(unigramModelTed.model, key=unigramModelTed.model.get)[0] # use word with highest probability as seed
    generated_text = ngramModelTed.generate_text(500, seed)
    write_document("ted.out.txt", generated_text)
    test_ted_out_dataset = prep_data("ted.out.txt")
    ppl = ngramModelTed.perplexity(test_ted_out_dataset)
    print("PPL using generated text 'ted.out.txt': " + str(round(ppl,4)))

    unigramModelReddit= NgramModel(dataset_reddit, len(dataset_vocab_reddit), 1, 0)
    seed = max(unigramModelReddit.model, key=unigramModelReddit.model.get)[0] # use word with highest probability as seed 
    # ?????? Since no punctuation were removed, "." ends up having the highest probability. The end of a line is most likely
    # The end of a line is most likely to be followed after period.
    # Because of this it ends up creating a file of just periods. 
    # To change this I just hardcoded the first word to 'i'. I still have the logic to use the max above
    seed = "i"
    generated_text = ngramModelReddit.generate_text(500, seed)
    write_document("reddit.out.txt", generated_text)
    test_reddit_out_dataset = prep_data("reddit.out.txt")
    ppl = ngramModelReddit.perplexity(test_reddit_out_dataset)
    print("PPL using generated text 'reddit.out.txt': " + str(round(ppl,4)))

    print("\n***Part 5 Done***\n")

if __name__ == "__main__":
    main()