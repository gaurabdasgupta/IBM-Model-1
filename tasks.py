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

def task_1(path):
    phrase_extraction_corpus_en = []
    phrase_extraction_corpus_fr = []
    words_en = set()
    words_fr = set()

    with open(path,'r') as f:
        d = json.load(f)

    for sent in d:
        phrase_extraction_corpus_en.append(sent['en'])
        phrase_extraction_corpus_fr.append(sent['fr'])
        fr_words = sent['fr'].split()
        en_words = sent['en'].split()
        for word in fr_words:
            words_fr.add(word)
        for word in en_words:
            words_en.add(word)
        
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
        ## Setting count(e|f) to 0 for all e,f
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

    return translation_prob


def task_2(path, alignments_pred):
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
        parallel_corpus.append(AlignedSent(fr_words, en_words))

    #  MODEL-1

    ibm1 = IBMModel1(parallel_corpus, 20)
    print("******IBM Model 1*******")
    for test in parallel_corpus:
        print("fr_sentence: {}".format(test.words))
        print("en_sentence: {}".format(test.mots))
        print("alignment: {}".format(test.alignment))
    #
    # for sent1 in parallel_corpus:
    #     for sent2 in parallel_corpus:
    #         for w in sent1.words:
    #             for t in sent2.mots:
    #                 print("test pairs: {}:{}".format(w, t))
    #                 print("actual: {}".format(ibm1.translation_table[w][t]))
    #                 print("predicted: {}".format(alignemnts_pred[t][w]))


    #  MODEL - 2

    # ibm2 = IBMModel2(parallel_corpus, 20)
    #
    # # print(ibm1.translation_table['maison']['house'])
    # print("******2*******")
    # for test in parallel_corpus:
    #     print("fr_sentence: {}".format(test.words))
    #     print("en_sentence: {}".format(test.mots))
    #     print("alignment: {}".format(test.alignment))

    return parallel_corpus, phrase_extraction_corpus_en, phrase_extraction_corpus_fr


def task_3(parallel_corpus, phrase_extraction_corpus_en, phrase_extraction_corpus_fr):

    alignments = []

    for sent_pair in parallel_corpus:
        # print(sent_pair.words)
        # print(sent_pair.alignment)
        alignment = []
        al = sent_pair.alignment
        for s in al:
            alignment.append(s)
        alignments.append(alignment)
    en_fr_phrases = []
    fr_phrases = []
    for i in range(len(phrase_extraction_corpus_en)):
        phrases = phrase_based.phrase_extraction(phrase_extraction_corpus_en[i], phrase_extraction_corpus_fr[i], alignments[i])
        for _, _, e_ph, f_ph in sorted(phrases):
            # print((e_ph,f_ph))
            # print(f_ph)
            en_fr_phrases.append((e_ph, f_ph))
            fr_phrases.append(f_ph)

    en_fr_phrases_count = Counter(en_fr_phrases)
    fr_phrases_count = Counter(fr_phrases)
    result = []

    for e, f in en_fr_phrases:
        result.append(((en_fr_phrases_count[(e, f)]/fr_phrases_count[f]), (e, f)))

    for i in reversed(sorted(set(result))):
        print(i)

    # print(en_fr_phrases)
    # print(fr_phrases)


if __name__ == "__main__":
    file_path = "./data/data2.json"
    #bitext, phrases_en, phrases_fr = task_2(file_path)
    #task_3(bitext, phrases_en, phrases_fr)
    alignemnts_pred = task_1(file_path)
    a, b, c = task_2(file_path, alignemnts_pred)
    # task_3(a, b, c)

    # for a in alignemnts:
    #     print(a + "  :  " + str(alignemnts[a]))
    #     sum = 0
    #     for b in alignemnts[a]:
    #         sum += alignemnts[a][b]
    #     print("sum : " + str(sum))
    #     print()