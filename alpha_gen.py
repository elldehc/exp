import random
import pickle


random.seed(114514)
alpha=[]
for i in range(100000):
    alpha.append((random.random(),random.random(),random.random()))
pickle.dump(alpha,open("our_alpha.pkl","wb"))
