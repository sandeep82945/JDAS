from pickletools import read_long4
import torch
from sentence_transformers import SentenceTransformer
import nltk


import pickle


print("ASPECT_SENTIMENT_DECISION")
path="/home/sandeep_2121cs29/hardik/COLING_2022/ckpt/aspect_importance_asp_senti_deci_best.pickle"

with open (path,'rb') as out:
  a=pickle.load(out).detach().cpu()

print(a)


print("ASPECT_DECISION")
path="/home/sandeep_2121cs29/hardik/COLING_2022/ckpt/aspect_importance_asp_deci_best.pickle"

with open (path,'rb') as out:
  a=pickle.load(out).detach().cpu()

print(a)