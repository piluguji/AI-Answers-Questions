import nltk
import sys
import os
import string
import math

from collections import Counter
from pprint import pprint

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")
    

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    #files = load_files("corpus")
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)
    
    askingQuestion = True
    while askingQuestion:
        # Prompt user for query
        query = set(tokenize(input("Query: ")))
        print("\n")

        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)
        
        askingQuestion = input("Another Question Y/N") == 'Y'


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    fileDict = {}
    for fileName in os.listdir(directory):
        fileContent = open(os.path.join(directory, fileName), "r", encoding="utf8")
        fileDict[fileName] = fileContent.read()
        fileContent.close()

    return fileDict

    raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    wordList = nltk.word_tokenize(document.lower())
    finalList = wordList.copy()
    for item in wordList: 
        if item in string.punctuation or item in nltk.corpus.stopwords.words("english"):
            finalList.remove(item)

    return finalList

    raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    wordsInDocuments = {}
    for text in documents.values():
        wordsInText = countNumberOfWords(text)
        for word in wordsInText:
            if word not in wordsInDocuments:
                wordsInDocuments[word] = 1
            else: 
                wordsInDocuments[word] += 1  
                

    wordIDF = {}
    for word in wordsInDocuments: 
        wordIDF[word] = math.log(len(documents) / wordsInDocuments[word])

    return wordIDF

                

    raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    documentScore = {}
    for name,text in files.items():  
        sum_tf_idf = 0
        for word in query: 
            occurences = text.count(word)
            try:
                sum_tf_idf += occurences * idfs[word]
            except: 
                sum_tf_idf += 0
        
        documentScore[name] = sum_tf_idf

    ranks = sorted(documentScore.items(), key=lambda x: x[1], reverse=True)
    
    #s = [[str(e) for e in row] for row in ranks]
    #lens = [max(map(len, col)) for col in zip(*s)]
    #fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    #table = [fmt.format(*row) for row in s]
    #print ('\n'.join(table))
    #print("\n")

    docRank = []
    for i in range(n):
        docRank.append(ranks[i][0])

    return docRank
    

    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentenceScore = {}
    for s, words in sentences.items(): 
        sum = 0
        count = 0
        for word in query:
            if word in words: 
                count += 1
                sum += idfs[word]
        sentenceScore[s] = [sum, count/len(words)]

    rankedSentences = sorted(sentenceScore.items(), key=lambda x: x[1], reverse=True)

    #s = [[str(e) for e in row] for row in rankedSentences[:5]]
    #lens = [max(map(len, col)) for col in zip(*s)]
    #fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    #table = [fmt.format(*row) for row in s]
    #print ('\n'.join(table))
    #print("\n")

    finalSentences = []
    for i in range(n):
        finalSentences.append(rankedSentences[i][0])

    return finalSentences

    raise NotImplementedError

def countNumberOfWords(text):
    wordsInText = {}
    for word in text: 
        if word not in wordsInText: 
            wordsInText[word] = text.count(word)

    return wordsInText

if __name__ == "__main__":
    main()
