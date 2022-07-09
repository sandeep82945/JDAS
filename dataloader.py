import torch
import json
import pickle
import torch
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk


#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:2")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")


model_name='bert-base-uncased'
tokenizer=AutoTokenizer.from_pretrained(model_name)

from torch.utils.data import Dataset, DataLoader

def onehot_aspect_sentiment(asp):
        # d={'motivation_positive':0,'clarity_positive':1,'soundness_positive':2,'substance_positive':3,'originality_positive':4,'meaningful_positive':5,'replicability_positive':6,
        #    'motivation_negative':7,'clarity_negative':8,'soundness_negative':9,'substance_negative':10,'originality_negative':11,'meaningful_negative':12,'replicability_negative':13}
        
        #d={'motivation':0,'clarity':1,'soundness':2,'substance':3,'originality':4,'meaningful':5,'replicability':6}
        d={'motivation':0, 'clarity':1, 'soundness':2, 'substance':3, 'meaningful':4, 'originality':5, 'replicability':6, 'no_aspect':7}
        
        d_sent={'positive':0,'negative':1}
  
        out1=torch.zeros(7)
        out2=torch.zeros(7,2)

        for a in asp:
            aspect,sentiment=a.split('_')
            out1[d[aspect]]=1
            out2[d[aspect]][d_sent[sentiment]]=1
       
        return out1,out2


class Data(Dataset):
    def __init__(self,review1,review2,review3,aspect1,aspect2,aspect3,decision):

        self.review1=review1
        self.review2=review2
        self.review3=review3
        
        self.aspect1=aspect1
        self.aspect2=aspect2
        self.aspect3=aspect3
        
        self.max_len=500
        self.decision=decision
        
        self.size=len(review1)

    @classmethod
    def getReader(cls,low,up,test=False,dummy=False):
        
        if(dummy):
            low=int(low/5)
            up=int(up/5)

        if(test==False):
          with open("data/created_files/dataframe_coling_reviews.json",'rb') as out:
              data= json.load(out)
              
              review1=[]
              aspect1=[]

              review2=[]
              aspect2=[]

              review3=[]
              aspect3=[]

              decision=[]

              paper_ids=list(data.keys())[low:up]
              
              for p in paper_ids:
                  
                  if(data[p]['decision'] is None or len(data[p]['Reviews'])<3):
                      continue
                  
                  
                  r1=list(data[p]['Reviews'].keys())[0]
                  a1=data[p]['Reviews'][r1]
                  
                  review1.append(r1)
                  aspect1.append(a1)

                  if(len(data[p]['Reviews'])>1):

                      r2=list(data[p]['Reviews'].keys())[1]
                      a2=data[p]['Reviews'][r2]

                      review2.append(r2)
                      aspect2.append(a2)
                      

                  else:
                      review2.append("")
                      aspect2.append([])
                      continue

                  if (len(data[p]['Reviews'])>2):
                      r3=list(data[p]['Reviews'].keys())[2]
                      a3=data[p]['Reviews'][r3]

                      review3.append(r3)
                      aspect3.append(a3)
                      
                  else:
                      review3.append("")
                      aspect3.append([])
                      continue
                  
                  decision.append(data[p]['decision'])

        #print(aspect1)
        return cls(review1,review2,review3,aspect1,aspect2,aspect3,decision)
    

    def __getitem__(self,idx):

        r1=self.review1[idx]
        r2=self.review2[idx]
        r3=self.review3[idx]

        de=self.decision[idx]
        des=torch.zeros(1)

        if(de=='accept'):
            des[0]=1
            

        a1,s1=onehot_aspect_sentiment(self.aspect1[idx])
        a2,s2=onehot_aspect_sentiment(self.aspect2[idx])
        a3,s3=onehot_aspect_sentiment(self.aspect3[idx])
        

        inputs = tokenizer(r1,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_length = True,
            truncation=True)
        
        ids1 = inputs['input_ids']
        mask1 = inputs['attention_mask']
        token_type_ids1 = inputs['token_type_ids']

        inputs = tokenizer(r2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_length = True,
            truncation=True)
        
        ids2 = inputs['input_ids']
        mask2 = inputs['attention_mask']
        token_type_ids2 = inputs['token_type_ids']

        inputs = tokenizer(r3,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_length = True,
            truncation=True)
        
        ids3 = inputs['input_ids']
        mask3 = inputs['attention_mask']
        token_type_ids3 = inputs['token_type_ids']

        
        return {
            'ids1': torch.tensor(ids1),
            'mask1': torch.tensor(mask1),
            'token_type_ids1': torch.tensor(token_type_ids1, dtype=torch.float64),
            'targets1':a1,
            'targets_senti1':s1,
            'ids2': torch.tensor(ids2),
            'mask2': torch.tensor(mask2),
            'token_type_ids2': torch.tensor(token_type_ids2, dtype=torch.float64),
            'targets2':a2,
            'targets_senti2':s2,
            'ids3': torch.tensor(ids3),
            'mask3': torch.tensor(mask3),
            'token_type_ids3': torch.tensor(token_type_ids3, dtype=torch.float64),
            'targets3':a3,
            'targets_senti3':s3,
            'decision':des}
        
    def __len__(self):
        return self.size

def getLoaders (batch_size):
       
        #dummy=True
        dummy=False
        print('Reading the training Dataset...')
        train_dataset = Data.getReader(0,7200,dummy=dummy) #19200 #21216
        
        print('Reading the validation Dataset...')
        valid_dataset = Data.getReader(7200,7700,dummy=dummy) #23200 #25216

        print('Reading the test Dataset...')
        test_dataset = Data.getReader(7700,8742,dummy=dummy) #23200:25248
        
        trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, num_workers=2,shuffle=True)
        validloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, num_workers=2,shuffle=True)
        testloader = DataLoader(dataset=test_dataset, batch_size = batch_size, num_workers=2)
        
        return trainloader, validloader, testloader


if __name__=='__main__':
    trainloader, validloader, testloader = getLoaders(batch_size=4)

    print("Length of TrainLoader:",len(trainloader))
    print("Length of ValidLoader:",len(validloader))
    print("Length of TestLoader:",len(testloader))

    for k in trainloader:
        print(k['ids1'].size())
        print(k['ids2'].size())
        print(k['ids3'].size())
        print(k['targets1'].size())
        print(k['targets2'].size())
        print(k['targets3'].size())
        print(k['targets_senti1'].size())
        break