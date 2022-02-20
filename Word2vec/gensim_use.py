import json
from word2vec.py import *

class country_Capital():

    def __init__(self,alist, wordvec):
        """Tuple is a pair of correct capital and country"""
        self.pair = alist
        self.word_vec = wordvec

        with open('relations.json') as instream:
            rel = json.load(instream)

        self.cities = rel['capital-common-countries'][0]
        self.countries = rel['capital-common-countries'][1]

    def predict_capital(self,alistofcountries):
        """Predicts the capital of countries in the list using the relation in the list"""
        
        pred_capital = [self.word_vec.most_similar(positive = self.pair, negative = city) for city in alistofcountries]

        return pred_capital,alistofcountries
    
    def predict_country(self,alistofcapitals):
        """Predicts the capital of countries in the list using the relation in the list"""
        
        pred_country= [self.word_vec.most_similar(positive = self.pair, negative = city) for city in alistofcapitals]

        return alistofcapitals,pred_country

    def accuracy(self, method = "capital"):

        scores = []
        
        if method == "capital":
            pred_capital,_ = self.predict_capital(self.countries)
            for capital,actual_capital  in zip(pred_capital,self.cities):
                if capital == actual_capital:
                        scores.append(1)
                else:
                        scores.append(0)
                
        elif method == "country": 
            _,pred_country = self.predict_country(self.cities)
            for country,goldcountry  in zip(pred_country, self.countries):
                if country == goldcountry:
                    scores.append(1)
                else:
                    scores.append(0)
                
        else:
            print("Not implemented".format(method))
         
        return sum(scores)/len(scores)

    def assess(self):

        self.predict_capital()
        accuracy_capital = self.accuracy(method = "capital")
        self.predict_country
        accuracy_country = self.accuracy(method = "country")

        return accuracy_capital, accuracy_country
    
