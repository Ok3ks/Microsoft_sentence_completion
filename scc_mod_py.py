# -*- coding: utf-8 -*-


import csv
from msilib.schema import SelfReg
import random
from wsgiref import headers

from click import option
from language_model_py import *
from operator import itemgetter
from nltk import word_tokenize

parentdir = "sentence-completion"
questions = "Microsoft_sentence_completion\sentence-completion\_testing_data.csv"
answers =   "Microsoft_sentence_completion\sentence-completion\_test_answer.csv"


class question():
    
    def __init__(self,aline):
        self.fields=aline
    
    def get_field(self,field):
        return self.fields[question.colnames[field]]
    
    def add_answer(self,fields):
        self.answer=fields[1]
   
    def chooseA(self):
        return("a")

    def tokenize(self):
        return [word_tokenize(self.get_field('question'))]

    def get_left_context(self,sent_tokens,window = 1, target = "_____"):
      """Get words preceeding a target word"""
      
      found=-1

      for i,token in enumerate(sent_tokens):
        if token == target:
            found = i
            break
        if found >-1:
          return sent_tokens[i-window: i]
        else: 
          return []

    def get_right_context(self,sent_tokens,window = 1, target = "_____"):
      """Get words preceeding a target word"""
      
      found=-1

      for i,token in enumerate(sent_tokens):
        if token == target:
            found = i
            break
        if found >-1:
          return sent_tokens[i: window+i]
        else: 
          return []

    def predict(self,method="chooseA"):

        choice = ["a)","b)","c)","d)", "e)"]
        methoda = {"method" : "unigram", "smoothing":"kneser_ney" }
        #eventually there will be lots of methods to choose from
        #current_question = question(scc_reader.q)
        #q = question()

        headers = [str(ind) for ind,_ in question.colnames.items()]

        n = len(self.fields)

        choice_answers = [self.get_field(i) for i in headers[2:]]
        #n = self.get_field('id')
        

        if method=="chooseA":
            return self.chooseA()
        else: 
            left_word = self.get_left_context(self.tokenize())
            right_word = self.get_right_context(self.tokenize())
            #question_number = int(self.get_field('id')) - 1  #subtracting header column
            #n = int(scc_reader.question.get_field('id')) - 1

            if method == "random": 
                choosen_option = random.randint(0,4)
                return choice_answers[choosen_option]

            if method == "unigram":
                probs = [(opt, mylm.get_prob(opt, methodparams = methoda)) for opt in choice_answers]     
                return sorted(probs, key = operator.itemgetter(1), reverse = True)[0][0]
            if method == "bigram_l":
                probs = [(opt, mylm.get_prob(left_word, context = opt, methodparams = methoda)) for opt in choice_answers]
                return sorted(probs, key = operator.itemgetter(1), reverse = True)[0][0]
            if method == "bigram_r":
                probs = [(opt, mylm.get_prob(opt, context = right_word, methodparams = methoda)) for opt in choice_answers]
                return sorted(probs, key = operator.itemgetter(1), reverse = True)[0][0]
            if method == "bigram":
                probs = [(opt,  mylm.get_prob(opt, context = right_word, methodparams = methoda) * mylm.get_prob(left_word, context = opt, methodparams = methoda)) for opt in choice_answers]
                return sorted(probs, key = operator.itemgetter(1), reverse = True)[0][0]

# get scc_mod working then Add triagrams in language model and


    def predict_and_score(self,method="chooseA"):
        
        #compare prediction according to method with the correct answer
        #return 1 or 0 accordingly
        prediction=self.predict(method=method)
        if prediction ==self.answer:
            return 1
        else:
            return 0

class scc_reader():
    
    def __init__(self,qs=questions,ans=answers):
        self.qs=qs
        self.ans=ans
        self.read_files()
        
    def read_files(self):
        
        #read in the question file
        with open(self.qs) as instream:
            csvreader=csv.reader(instream)
            self.qlines=list(csvreader)
        
        #store the column names as a reverse index so they can be used to reference parts of the question
        question.colnames={item:i for i,item in enumerate(self.qlines[0])}
        #id = 0, question = 1, a = 2, b =3, c =4, d =5, e=6
        
        #create a question instance for each line of the file (other than heading line)
        scc_reader.questions=[question(qline) for qline in self.qlines[1:]]
        
        #read in the answer file
        with open(self.ans) as instream:
            csvreader=csv.reader(instream)
            alines=list(csvreader)
            
        #add answers to questions so predictions can be checked    
        for q,aline in zip(self.questions,alines[1:]):
            q.add_answer(aline)
        
    def get_field(self,field):
        return [q.get_field(field) for q in scc_reader.questions] 
    
    def predict(self,method="chooseA"):
        return [q.predict(method=method) for q in scc_reader.questions]
    
    def predict_and_score(self,method="chooseA"):
        scores=[q.predict_and_score(method=method) for q in scc_reader.questions]
        return sum(scores)/len(scores)


#SCC = scc_reader()
#question = question(SCC.qlines[1])

#print(question.predict(method = 'unigram'))

#print(question.predict_and_score(method = 'unigram'))