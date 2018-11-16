from __future__ import division
from collections import defaultdict
from nltk.translate import AlignedSent
from nltk.translate import Alignment
from nltk.translate import IBMModel
from nltk.translate.ibm_model import Counts
import warnings
from nltk.translate import IBMModel1
from nltk.translate import IBMModel2
import json
from nltk.translate import phrase_based

parallel_corpus = []
phrase_extraction_corpus_en = []
phrase_extraction_corpus_fr = []

path = "./data/data2.json"

with open(path,'r') as f:
    d = json.load(f)

for sent in d:
    phrase_extraction_corpus_en.append(sent['en'])
    phrase_extraction_corpus_fr.append(sent['fr'])
    fr_words = sent['fr'].split()
    en_words = sent['en'].split()
    parallel_corpus.append(AlignedSent(fr_words, en_words))


#  MODEL-1
ibm1 = IBMModel1(parallel_corpus, 5)
# phrases = phrase_based.phrase_extraction()
alignments = []
# print("******1*******")
for sent_pair in parallel_corpus:
    # print(sent_pair.words)
    # print(sent_pair.alignment)
    alignment = []
    al = sent_pair.alignment
    for s in al:
        # s = str(s).replace(' ', '')
        # s = str(s).replace("'","")
        alignment.append(s)
    alignments.append(alignment)

# print(alignments)
# f_aligned = [j for _,j in alignments]
# print(f_aligned)

for i in range(len(phrase_extraction_corpus_en)):
    phrases = phrase_based.phrase_extraction(phrase_extraction_corpus_en[i], phrase_extraction_corpus_fr[i], alignments[i])
    for i in sorted(phrases):
        print(i)

# MODEL-2
# ibm2 = IBMModel2(parallel_corpus, 20)
#
# # print(ibm1.translation_table['maison']['house'])
# print("******2*******")
# for test in parallel_corpus:
#     print("fr_sentence: {}".format(test.words))
#     print("en_sentence: {}".format(test.mots))
#     print("alignment: {}".format(test.alignment))
