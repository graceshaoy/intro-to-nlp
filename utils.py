import pandas as pd
import numpy as np
import altair as alt
from sklearn.manifold import TSNE

import collections
import sys

import regex as re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

### LIWC ######################################
def readDict(dictionaryPath):
    '''
    Function to read in an LIWC-style dictionary
    '''
    catList = collections.OrderedDict()
    catLocation = []
    wordList = {}
    finalDict = collections.OrderedDict()

    # Check to make sure the dictionary is properly formatted
    with open(dictionaryPath, "r") as dictionaryFile:
        for idx, item in enumerate(dictionaryFile):
            if "%" in item:
                catLocation.append(idx)
        if len(catLocation) > 2:
            # There are apparently more than two category sections;
            # throw error and die
            sys.exit("Invalid dictionary format.")

    # Read dictionary as lines
    with open(dictionaryPath, "r") as dictionaryFile:
        lines = dictionaryFile.readlines()

    # Within the category section of the dictionary file, grab the numbers
    # associated with each category
    for line in lines[catLocation[0] + 1:catLocation[1]]:
        catList[re.split(r'\t+', line)[0]] = [re.split(r'\t+',
                                                       line.rstrip())[1]]

    # Now move on to the words
    for idx, line in enumerate(lines[catLocation[1] + 1:]):
        # Get each line (row), and split it by tabs (\t)
        workingRow = re.split('\t', line.rstrip())
        wordList[workingRow[0]] = list(workingRow[1:])

    # Merge the category list and the word list
    for key, values in wordList.items():
        if not key in finalDict:
            finalDict[key] = []
        for catnum in values:
            workingValue = catList[catnum][0]
            finalDict[key].append(workingValue)
    return (finalDict, catList.values())

def wordCount(data, dictOutput):
    '''
    Function to count and categorize words based on an LIWC dictionary
    '''
    finalDict, catList = dictOutput

    # Create a new dictionary for the output
    outList = collections.OrderedDict()

    # Number of non-dictionary words
    nonDict = 0

    # Convert to lowercase
    data = data.lower()

    # Tokenize and create a frequency distribution
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(data)

    fdist = nltk.FreqDist(tokens)
    wc = len(tokens)

    # Using the Porter stemmer for wildcards, create a stemmed version of data
    porter = nltk.PorterStemmer()
    stems = [porter.stem(word) for word in tokens]
    fdist_stem = nltk.FreqDist(stems)

    # Access categories and populate the output dictionary with keys
    for cat in catList:
        outList[cat[0]] = 0

    # Dictionaries are more useful
    fdist_dict = dict(fdist)
    fdist_stem_dict = dict(fdist_stem)

    # Number of classified words
    classified = 0

    for key in finalDict:
        if "*" in key and key[:-1] in fdist_stem_dict:
            classified = classified + fdist_stem_dict[key[:-1]]
            for cat in finalDict[key]:
                outList[cat] = outList[cat] + fdist_stem_dict[key[:-1]]
        elif key in fdist_dict:
            classified = classified + fdist_dict[key]
            for cat in finalDict[key]:
                outList[cat] = outList[cat] + fdist_dict[key]

    # Calculate the percentage of words classified
    if wc > 0:
        percClassified = (float(classified) / float(wc)) * 100
    else:
        percClassified = 0

    # Return the categories, the words used, the word count,
    # the number of words classified, and the percentage of words classified.
    return [outList, tokens, wc, classified, percClassified]

def liwc_features(text, liwc_dict, liwc_categories):
    '''
    Compute rel. percentage of LIWC 2007 categories:
    'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family',
    'friend
    '''
    liwc_counts = wordCount(text, liwc_dict)

    return [liwc_counts[0][cat] / liwc_counts[2] for cat in liwc_categories].sum()

### LDA ########################################
def fill_topic_weights(df_row, bow_corpus, ldamodel):
    '''
    Fill DataFrame rows with topic weights for topics in songs.

    Modifies DataFrame rows *in place*.
    '''
    try:
        for i in ldamodel[bow_corpus[df_row.name]]:
            df_row[str(i[0])] = i[1]
    except:
        return df_row
    return df_row

def doc2vec_tsne(doc_model, perplexity=40, n_iter=2500, n_components=2):
    tokens = []
    for i in range(len(doc_model.dv.vectors)):
        tokens.append(doc_model.dv.vectors[i])

    # Reduce n dimensional vectors down into 2-dimensional space
    tsne_model = TSNE(perplexity=perplexity, n_components=n_components, init='pca',
                      n_iter=n_iter, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    df = pd.DataFrame()
    for i in range(n_components):
        df['X'+str(i+1)] = [doc[i] for doc in new_values]

    return df

### Sentiment Analysis ########################################
def sentiment_analyzer_list(doclist):
    
    analyser  = SentimentIntensityAnalyzer()
    pos, neg, neut  = [], [], []
    
    for doc in doclist:
        score = analyser.polarity_scores(doc)
        
        if score['compound']   >0: # positive comments
            pos.append(doc)
        elif score['compound'] <0: # negative comments
            neg.append(doc)
        else:
            neut.append(doc)
    return (pos, neut, neg)

def sentiment_analyzer_sin(doc):
    
    analyser  = SentimentIntensityAnalyzer()
    
    score = analyser.polarity_scores(doc)
    return score['compound']