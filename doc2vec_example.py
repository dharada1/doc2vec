#coding: UTF-8
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

m = Doc2Vec.load('model/doc2vec.model')

#文書0と似てるやつが順番に出てくる
print "similar to no.0:"
print m.docvecs.most_similar(0)

#文書1と80の類似度
print "similarity:"
print m.docvecs.similarity(1,80)

