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


def task_2(path):
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
    print("******1*******")
    # for test in parallel_corpus:
    #     print("fr_sentence: {}".format(test.words))
    #     print("en_sentence: {}".format(test.mots))
    #     print("alignment: {}".format(test.alignment))

    # MODEL-2

    # ibm2 = IBMModel2(parallel_corpus, 20)

    # print(ibm1.translation_table['maison']['house'])
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
            en_fr_phrases.append((e_ph,f_ph))
            fr_phrases.append(f_ph)

    en_fr_phrases_count = Counter(en_fr_phrases)
    fr_phrases_count = Counter(fr_phrases)
    result = []

    for e, f in en_fr_phrases:
        result.append(((en_fr_phrases_count[(e, f)]/fr_phrases_count[f]), (e, f)))

    for i in reversed(sorted(result)):
        print(i)

    # print(en_fr_phrases)
    # print(fr_phrases)


if __name__ == "__main__":
    file_path = "./data/data2.json"
    bitext, phrases_en, phrases_fr = task_2(file_path)
    task_3(bitext, phrases_en, phrases_fr)
