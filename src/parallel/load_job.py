import numpy as np
from gensim.models import Word2Vec
import logging as logger
import time
import cython_knn as knn
path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'
logger.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logger.INFO)

model = Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
result = np.zeros([3000000],dtype=np.float32)

start = time.time()
knn.compute_knn(vectors = model.syn0,result = result,x=3000000,y=300)
end = time.time()
logger.info("Time for computation: " + str(end-start)) 
for i in range(0,10):
	logger.info(result[i])

