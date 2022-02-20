from scc_mod_py import *

SCC = scc_reader()
init_questions = SCC.questions

methods = ["","random","unigram","bigram_l","bigram_r","bigram"]

#for a in methods:  
    #print("The score is {}".format(SCC.predict_and_score(method = a)))

print("The score is {}".format(SCC.predict_and_score(method = "unigram")))
