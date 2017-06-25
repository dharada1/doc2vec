#coding: UTF-8
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

f = open('articles.100.owakati.txt','r')#空白で単語を区切り、改行で文書を区切っているテキストデータ

#１文書ずつ、単語に分割してリストに入れていく[([単語1,単語2,単語3],文書id),...]こんなイメージ
#words：文書に含まれる単語のリスト（単語の重複あり）
# tags：文書の識別子（リストで指定．1つの文書に複数のタグを付与できる）
trainings = [TaggedDocument(words = data.split(),tags = [i]) for i,data in enumerate(f)]


# トレーニング（パラメータについては後日）
m = Doc2Vec(documents= trainings, dm = 1, size=300, window=8, min_count=10, workers=4)

# モデルのセーブ
m.save("model/doc2vec.model")
