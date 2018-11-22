from __future__ import division
from collections import defaultdict
from collections import Counter
from nltk.translate import AlignedSent
from nltk.translate import Alignment
from nltk.translate import IBMModel
from nltk.translate.ibm_model import Counts
import warnings
from nltk.translate import IBMModel1
from nltk.translate import IBMModel2
import json
from nltk.translate import phrase_based
import time
import sys

def task_1(path):
    """
    Task 1: Implemented the IBM Model 1 and EM Algorithm
    :param path: path for data
    :return: list of tuples: alignments and execution time
    """
    phrase_extraction_corpus_en = []
    phrase_extraction_corpus_fr = []
    words_en = set()
    words_fr = set()
    # read file
    with open(path,'r') as f:
        d = json.load(f)

    start = time.process_time()
    # store words in list
    for sent in d:
        phrase_extraction_corpus_en.append(sent['en'])
        phrase_extraction_corpus_fr.append(sent['fr'])
        fr_words = sent['fr'].split()
        en_words = sent['en'].split()
        for word in fr_words:
            words_fr.add(word)
        for word in en_words:
            words_en.add(word)
    # compute probabilities
    words_en = list(words_en)
    words_fr = list(words_fr)
    num_fr = len(words_fr)
    translation_prob = dict()
    for en_word in words_en:
        translation_prob[en_word] = dict()
        for fr_word in words_fr:
            translation_prob[en_word][fr_word] = 1 / num_fr

    # Till convergence part here
    num_iterations = 100
    for i in range(num_iterations):

        # Setting count(e|f) to 0 for all e,f
        count = dict()
        for en_word in words_en:
            count[en_word] = dict()
            for fr_word in words_fr:
                count[en_word][fr_word] = 0
        
        #  Setting up total(f) = 0 for all f
        total = dict()
        for fr_word in words_fr:
            total[fr_word] = 0
        
        # For all sentence pairs
        numSentencePairs = len(phrase_extraction_corpus_en)
        for j in range(numSentencePairs):
            en_sent = phrase_extraction_corpus_en[j]
            en_sent = en_sent.split()
            fr_sent = phrase_extraction_corpus_fr[j]
            fr_sent = fr_sent.split()

            # Computing Normalisation
            s_total = dict() 
            for e in en_sent:
                s_total[e] = 0
                for f in fr_sent:
                    s_total[e] += translation_prob[e][f]
            
            # Collecting counts
            for e in en_sent:
                for f in fr_sent:
                    count[e][f] += translation_prob[e][f] / s_total[e]
                    total[f] += translation_prob[e][f] / s_total[e]
        
        # Estimate probabilities
        for f in words_fr:
            for e in words_en:
                translation_prob[e][f] = count[e][f] / total[f]

        # Forming alignments
        alignments = dict()
        for sentIndex in range(len(phrase_extraction_corpus_en)):
            en_sent = phrase_extraction_corpus_en[sentIndex]
            dict_key = en_sent
            fr_sent = phrase_extraction_corpus_fr[sentIndex]

            en_sent = en_sent.split()
            fr_sent = fr_sent.split()

            align = []
            for en_index in range(len(en_sent)):
                max_fr_index = 0
                enWord = en_sent[en_index]
                for fr_index in range(len(fr_sent)):
                    frWord = fr_sent[fr_index]
                    probMax = translation_prob[enWord][fr_sent[max_fr_index]]
                    if probMax < translation_prob[enWord][frWord]:
                        max_fr_index = fr_index
                align.append((en_index, max_fr_index))

            alignments[dict_key] = align
    end = time.process_time()
    exec_time = str(end-start)

    return alignments, exec_time


def task_2(path, alignments_pred):
    """
    Task 2: Comparing our alignment results with that of NLTK library's output of IBM Model 1 and IBM Model 2
    :param path: path for data
    :param alignments_pred: alignments computed in task 1
    :return: parallel_corpus, phrase_extraction_corpus_en, phrase_extraction_corpus_fr
    """
    parallel_corpus = []
    phrase_extraction_corpus_en = []
    phrase_extraction_corpus_fr = []
    with open(path,'r') as f:
        d = json.load(f)

    for sent in d:
        phrase_extraction_corpus_en.append(sent['en'])
        phrase_extraction_corpus_fr.append(sent['fr'])
        fr_words = sent['fr'].split()
        en_words = sent['en'].split()
        parallel_corpus.append(AlignedSent(en_words, fr_words))


    # MODEL - 2

    print("******IBM Model-2*******")

    ibm2 = IBMModel2(parallel_corpus, 50)
    for test in parallel_corpus:
        print("en_sentence: {}".format(test.words))
        print("fr_sentence: {}".format(test.mots))
        try:
            print("nltk alignment: {}".format(test.alignment))
        except:
            print("nltk ibm model 2 alignment failed")

    #  MODEL-1

    ibm1 = IBMModel1(parallel_corpus, 50)
    print("******IBM Model 1*******")
    for test in parallel_corpus:
        print("en_sentence: {}".format(test.words))
        print("fr_sentence: {}".format(test.mots))
        try:
            print("nltk alignment: {}".format(test.alignment))
        except:
            print("nltk ibm model 1 alignment failed")
        str_test = ' '.join(word for word in test.words)
        print("predicted alignment: {}\n".format(alignemnts_pred[str_test]))

    return parallel_corpus, phrase_extraction_corpus_en, phrase_extraction_corpus_fr


def task_3(parallel_corpus, phrase_extraction_corpus_en, phrase_extraction_corpus_fr, alignments_pred):
    """
    Task 3: Utility for calculating phrase based extraction scoring
    :param parallel_corpus: Processed Bitext
    :param phrase_extraction_corpus_en
    :param phrase_extraction_corpus_fr
    :param alignments_pred: Alignment list computed in task 1
    :return: execution time
    """

    start = time.process_time()
    alignments = []
    print("Phrase Extraction")

    en_fr_phrases = []
    fr_phrases = []
    # print(phrase_extraction_corpus_en)
    for i in range(len(phrase_extraction_corpus_en)):
        # print(alignments_pred[phrase_extraction_corpus_en[i]])
        # print(alignments[i])
        # srctext = "michael assumes that he will stay in the house"
        # trgtext = "michael geht davon aus , dass er im haus bleibt"
        # alignment = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 5), (3, 6), (4, 9), (5, 9), (6, 7), (7, 7), (8, 8)]
        # phrases = phrase_based.phrase_extraction(srctext, trgtext, alignment)
        # print(phrase_extraction_corpus_en[i])
        phrases = phrase_based.phrase_extraction(phrase_extraction_corpus_en[i], phrase_extraction_corpus_fr[i], alignments_pred[phrase_extraction_corpus_en[i]])
        # print("here")
        # for i in sorted(phrases):
        #     print(i)
        for _, _, e_ph, f_ph in sorted(phrases):
            en_fr_phrases.append((e_ph, f_ph))
            fr_phrases.append(f_ph)

    en_fr_phrases_count = Counter(en_fr_phrases)
    fr_phrases_count = Counter(fr_phrases)
    result = []
    # print(en_fr_phrases_count)
    # print(fr_phrases_count)
    for e, f in en_fr_phrases:
        result.append(((en_fr_phrases_count[(e, f)]/fr_phrases_count[f]), (e, f)))

    for i in reversed(sorted(set(result))):
        print(i)

    end = time.process_time()
    exec_time = str(end-start)

    return exec_time


if __name__ == "__main__":
    file_path = str(sys.argv[1])
    alignemnts_pred, t1_time = task_1(file_path)
    print("task 1 CPU time: {}".format(t1_time))
    parallel_corpus, phrase_extraction_corpus_en, phrase_extraction_corpus_fr = task_2(file_path, alignemnts_pred)
    t3_time = task_3(parallel_corpus, phrase_extraction_corpus_en, phrase_extraction_corpus_fr, alignemnts_pred)
    print("task 3 CPU time : {}".format(t3_time))
