from gensim.models import KeyedVectors

filename = "\wk1labresources\GoogleNews-vectors-negative300"

vec = KeyedVectors.load_word2vec_format(filename, binary = True )