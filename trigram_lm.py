
from itertools import count
from nltk import word_tokenize as tokenize
import operator
import os,random,math
import re

parentdir = "drive/My Drive/ML/text-data/sentence-completion"
TRAINING_DIR= parentdir + "/Holmes_Training_Data"

def get_training_testing(training_dir=TRAINING_DIR,split=0.5):

    filenames=os.listdir(training_dir)
    n=len(filenames)
    print("There are {} files in the training directory: {}".format(n,training_dir))
    #random.seed(53)  #if you want the same random split every time
    random.shuffle(filenames)
    index=int(n*split)
    return(filenames[:index],filenames[index:])

trainingfiles,heldoutfiles=get_training_testing()


class language_model():
    
    def __init__(self,trainingdir=TRAINING_DIR,files=[]):
        self.training_dir=trainingdir
        self.files=files 
        self.unigram={}
        self.bigram={}
        self.trigram = {}   

    #My code 

    def display_bigram(self):
        return self.bigram
    def display_unigram(self):
        return self.unigram
    def display_trigram(self):
        return self.trigram


        
    
    def _processline(self,line):

        tokens=["__START"]+tokenize(line)+["__END"]
        previous="__END"

        for count,token in enumerate(tokens):

            #processing unigram
            self.unigram[token]=self.unigram.get(token,0)+1

            #processing bigram
            current = self.bigram.get(token, {})   
            current[previous] = current.get(previous, 0)+1
            self.bigram[token] = current
                
            #processing trigram
            current_3 = self.trigram.get(previous, {})
            interm_ = current_3.get(token, {})
            
            if count + 1 < len(tokens) - 1: 
                next = tokens[count+1]
            else:
                next = " "
            
            interm_[next] = interm_.get(next,0)+1
            current_3[token] = interm_
            self.trigram[previous] = current_3

            previous=token
        
            #previous=token
        #for trigrams
            #if count < len(tokens) - 2: 
            #current_3 = self.trigram.get(token, {})
                #partner_3 =" ".join(tokens[count+1: count+3])#accounts for _END on each line 
                #current_3[partner_3] = current_3.get(partner_3, 0) + 1
                #self.trigram[token] = current_3
            #else: 
                #pass
            
    
    def _processfiles(self):

        for afile in self.files:
            print("Processing {}".format(afile))
            try:
                with open(os.path.join(self.training_dir,afile)) as instream:
                    for line in instream:
                        line=line.rstrip()
                        if len(line)>0:
                            self._processline(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring rest of file".format(afile))
      
            
    def _convert_to_probs(self):
        
        self.unigram= {k:v/sum(self.unigram.values()) for (k,v) in self.unigram.items()}
        self.bigram={key:{k:v/sum(adict.values()) for (k,v) in adict.items()} for (key,adict) in self.bigram.items()}
        self.trigram={key_:{key:{k:v/sum(adict.values()) for k,v in adict.items()} for key,adict in adict_.items()} for key_,adict_ in self.trigram.items()}
        
    def _convert_to_probs_kn(self):
        self.kn={k:v/sum(self.kn.values()) for (k,v) in self.kn.items()}
        self.kn_b={k:v/sum(self.kn_b.values()) for (k,v) in self.kn_b.items()}

    def get_prob(self,token,context="",methodparams={}):
        if methodparams.get("method","unigram")=="unigram":
            return self.unigram.get(token,self.unigram.get("__UNK",0))
        else:
            if methodparams.get("smoothing","kneser-ney")=="kneser-ney":
                unidist=self.kn
            else:
                unidist=self.unigram


            bigram=self.bigram.get(context[-1],self.bigram.get("__UNK",{}))
            big_p=bigram.get(token,bigram.get("__UNK",0))

            if methodparams.get("smoothing")=="none":
                p = big_p
            else:
                lmbda=bigram["__DISCOUNT"]
                uni_p=unidist.get(token,unidist.get("__UNK",0))
            #print(big_p,lmbda,uni_p)
                p=big_p+lmbda*uni_p          
            return p
    
    def get_prob_tri(self,previous,token,next , methodparams = {}):
        if methodparams.get("method", "trigram") == "trigram":
            first_layer = self.trigram.get(previous, self.trigram.get("__UNK",{}))
            second_layer = first_layer.get(token, first_layer.get("__UNK",{}))
            tri_p = second_layer.get(next, self.unigram.get("__UNK",0))
            
            p = tri_p
        else: 
            print("{} doesn't exist".format(methodparams))

        if methodparams.get("smoothing","kneser-ney")=="kneser-ney":
                
            bidist = self.kn_b #number of times bigram exists as novel combination

            lmbda_1 = second_layer["__DISCOUNT"]    
            big_p = bidist.get(token, bidist.get("__UNK", 0))

            unidist = self.kn   #number of occurence as novel combination
            uni_p = unidist.get(next) #for the last word in trigram

            temp = self.bigram.get(token, self.bigram.get("__UNK")) #using probability mass for token preceding last word of trigram
            lmbda_2 = temp["__DISCOUNT"]


            p = uni_p*lmbda_2 + big_p*lmbda_1 + tri_p
        else:
            pass            
        
        return p
    
    def nextlikely(self,k=1,current="",method="unigram"):
        blacklist=["__START","__DISCOUNT"]
       
        if method=="unigram":
            dist=self.unigram
        else:
            dist=self.bigram.get(current,self.bigram.get("__UNK",{}))
    
        mostlikely=list(dist.items())
        #filter out any undesirable tokens
        filtered=[(w,p) for (w,p) in mostlikely if w not in blacklist]
        #choose one randomly from the top k
        words,probdist=zip(*filtered)
        res=random.choices(words,probdist)[0]
        return res
    
    def generate(self,k=1,end="__END",limit=20,method="bigram",methodparams={}):
        if method=="":
            method=methodparams.get("method","bigram")
        current="__START"
        tokens=[]
        while current!=end and len(tokens)<limit:
            current=self.nextlikely(k=k,current=current,method=method)
            tokens.append(current)
        return " ".join(tokens[:-1])
    
    
    def compute_prob_line(self,line,methodparams={}):
        #this will add _start to the beginning of a line of text
        #compute the probability of the line according to the desired model
        #and returns probability together with number of tokens
        
        tokened = tokenize(line)
        tokens=["__START"]+tokened +["__END"]
        N = len(tokened)
        del tokened

        if methodparams.get("method") != "trigram":
            acc=0
            for i,token in enumerate(tokens[1:]):
                acc+=math.log(self.get_prob(token,tokens[:i+1],methodparams))
        else:
            acc=0; previous = "__START"
            for i,token in enumerate(tokens[1:]):
                acc+=math.log(self.get_prob_tri(previous,token,tokens[i+1],methodparams))
                previous = token

        return acc, N
    
    def compute_probability(self,filenames=[],methodparams={}):
        #computes the probability (and length) of a corpus contained in filenames
        if filenames==[]:
            filenames=self.files
        
        total_p=0
        total_N=0
        for i,afile in enumerate(filenames):
            print("Processing file {}:{}".format(i,afile))
            try:
                with open(TRAINING_DIR + "/" + afile ) as instream:
                    for line in instream:
                        line=line.rstrip()
                        if len(line)>0:
                            p,N=self.compute_prob_line(line,methodparams=methodparams)
                            total_p+=p
                            total_N+=N
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing file {}: ignoring rest of file".format(afile))
        return total_p,total_N
    
    def compute_perplexity(self,filenames=[],methodparams={"method":"bigram","smoothing":"kneser-ney"}):
        
        #compute the probability and length of the corpus
        #calculate perplexity
        #lower perplexity means that the model better explains the data
        
        p,N=self.compute_probability(filenames=filenames,methodparams=methodparams)
        #print(p,N)
        pp=math.exp(-p/N)
        return pp  
    
    def _make_unknowns(self,known=2):
        unknown=0
        
        #making unigram unknowns
        for (k,v) in list(self.unigram.items()):
            if v<known:
                del self.unigram[k]
                self.unigram["__UNK"]=self.unigram.get("__UNK",0)+v
        
        #making bigram unknowns
        for (k,adict) in list(self.bigram.items()):
            for (kk,v) in list(adict.items()):
                isknown=self.unigram.get(kk,0)
                if isknown<2:
                    adict["__UNK"]=adict.get("__UNK",0)+v
                    del adict[kk]
            
            isknown=self.unigram.get(k,0) #checking if the word is in the unigram
            
            if isknown<known:
                del self.bigram[k]
                current=self.bigram.get("__UNK",{})
                current.update(adict)
                self.bigram["__UNK"]=current
                
            else:
                self.bigram[k]=adict
        
        for (k,adict) in list(self.trigram.items()):
            for (kk,v) in list(adict.items()):
                for (kkk,vvv) in list(v.items()):
                    isknown=self.unigram.get(kkk,0)     #last word 
                    if isknown==0:
                        v["__UNK"]=v.get("__UNK",0)+vvv

                        del v[kkk]
            
                
                if kk not in self.bigram.keys(): #check main bigram if this particular bigram exists
                    del adict[kk]
                    current = self.bigram.get("__UNK",{})
                    current.update(v)
                    adict["__UNK"] = current
                else:
                    adict[kk] = v

            isknown = self.unigram.get(k,0)
            if isknown < known: #first word
                del self.trigram[k]
                current = self.trigram.get("__UNK",{})
                current.update(adict)
                self.trigram["__UNK"] = current
            else: 
                self.trigram[k] = adict

            
            #isknown=self.unigram.get(k,0)

            #if isknown==0:
                #del self.trigram[k]
                #current=self.trigram.get("__UNK",{})
                #current.update(adict)
                #self.trigram["__UNK"]=current
                    
            #else:
                #self.trigram[kk]=adict
                
    def _discount(self,discount=0.75):
        #discount each bigram count by a small fixed amount
        self.bigram={key:{k:v-discount for k,v in adict.items()} for key,adict in self.bigram.items()}
        self.trigram={key_:{key:{k:v - discount for (k,v) in adict.items()} for (key,adict) in adict_.items()} for (key_,adict_) in self.trigram.items()}
        
        #for each word, store the total amount of the discount so that the total is the same 
        #i.e., so we are reserving this as probability mass
        for k in self.bigram.keys():
            lamb=len(self.bigram[k])
            self.bigram[k]["__DISCOUNT"]=lamb*discount
        
        for k,v in self.trigram.items():
            for kk in v.keys():
                lamb_2 = len(v[kk])
                self.trigram[k][kk]["__DISCOUNT"] = lamb_2*discount
            
        #work out kneser-ney unigram probabilities
        #count the number of contexts each word has been seen in
        self.kn={}
        for (k,adict) in self.bigram.items():
            for kk in adict.keys():
                self.kn[kk]=self.kn.get(kk,0)+1

        #working out the kneser-ney bigram probabilities
        self.kn_b = {}
        for k,adict in self.trigram.items():
            for kk,adict2 in adict.items():
                for kkk in adict2.keys():
                    self.kn_b[kkk] = self.kn_b.get(kkk, 0) +1 #obtains count 

        
    def train(self):
    
        self._processfiles()
        self._make_unknowns()
        self._discount()
        self._convert_to_probs()
        self._convert_to_probs_kn()
    
    def train_2(self):
        
        self._processfiles()
        self._make_unknowns()
        self._convert_to_probs()

