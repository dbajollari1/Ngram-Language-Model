## Descriptions of code files
    - main.py: Driver file for calling functions and classes, and getting desired results
    - pre_process.py: Loads the data files and prepares them for processing  
    - ngram_model.py: A class that creates an ngram model of desired size, calulates perplexity on test files, and generates text

## Perplexity scores for uniform language model (part 1)
    - test.reddit.txt PPL: 42184.0343
    - test.ted.txt PPL: 58831.6641
    - test.news.txt PPL: 47721.4467

## Perplexity scores for unigram language model (part 2)
    - test.reddit.txt PPL: 1165.8393
    - test.ted.txt PPL: 600.1772
    - test.news.txt PPL: 1472.7726

## Plot for ted talk and reddit n-gram language models without smoothing (part 3)
![Part 3 Plot](Part3_Plot.png?raw=true "Part 3 Plot")

## Plot for ted talk and reddit n-gram language models with laplace smoothing (part 4)
![Part 4 Plot](Part4_Plot.png?raw=true "Part 4 Plot")

## Perplexity scores for the two word documents (part 5)
    - Perplexity score for ted.out.txt(used ngram size 7 with no laplace smoothing) - 1.0
    - Perplexity score for reddit.out.txt(used ngram size 7 with no laplace smoothing) - 1.0


*When genereating text for reddit.out.txt I ran into a problem. Since no punctuation were removed, "." ends up having the highest probability. The end of a line is most likely to be followed after period. Because of this it ends up creating a file of just periods. To change this I just hardcoded the first word to 'i'. I still have the logic to use the max probability in the code.*