import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import transformers
import datasets
import shap

# labels = classes = ['MOT', 'CLA', 'SOU','SUB', 'MEA', 'ORI', 'REP']


# #If there's a GPU available...
# if torch.cuda.is_available():    

#     # Tell PyTorch to use the GPU.    
#     device = torch.device("cuda:4")

#     print('There are %d GPU(s) available.' % torch.cuda.device_count())

#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# else:
#   print('No GPU available, using the CPU instead.')
#   device = torch.device("cpu")

# #torch.cuda.set_device(0)
# import json
# import numpy as np
# import os
# import random
# import re
# import pickle
# import torch
# from tqdm.autonotebook import tqdm
# from transformers import AutoTokenizer,AutoModel

# model_name='bert-base-uncased'

# tokenizer=AutoTokenizer.from_pretrained(model_name)

# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# import json
# import numpy as np
# import os
# import random
# import re
# import pickle
# import torch
# from tqdm.autonotebook import tqdm
# import torch
# from torch.utils.data import Dataset, DataLoader
# import nltk
# import torch.nn as nn
# import torch.nn.functional as F


# if torch.cuda.is_available():    
#     device = torch.device("cuda:4")

# else:
#    device = torch.device("cpu")

# torch.cuda.set_device(2)


# class BertModel(nn.Module):
#     def __init__(self, in_features, out_features):

#         super(BertModel, self).__init__()

#         self.in_features = in_features   #768
#         self.out_features = out_features    #7

#         self.flatten=nn.Flatten()
#         self.lstm_1 = nn.LSTM(in_features, 512//2, batch_first=True, bidirectional=True) #bidirectional=True

#         self.linear1=nn.Linear(512*7,512*2)
#         self.linear2=nn.Linear(512*2,256)
#         self.linear3=nn.Linear(256,64)

#         self.last_dense = nn.Linear(64, self.out_features)
#         self.dropout1=nn.Dropout(p=0.4)
#         self.dropout2=nn.Dropout(p=0.2)

#         self.relu = nn.ReLU()
#         self.sigmoid=nn.Sigmoid()

#         self.category=nn.Linear(512,out_features)
        
#         #SENTIMENT PART

#         self.linear_sen1=nn.Linear(512,256)
#         self.linear_sen2=nn.Linear(256,32)
#         self.linear_sen3=nn.Linear(32,2)
        
 
#         #DECSION PART OF THE MODEL       
#         hidden_dimension=512
    
#         self.linear_des1=nn.Linear(hidden_dimension*3,256)
#         self.linear_des2=nn.Linear(256,64)
#         self.linear_des3=nn.Linear(64,8)
#         self.linear_des4=nn.Linear(8,1)

#         imp_score= torch.rand( 1,7,requires_grad=True)  #(512,1)
#         nn.init.xavier_normal_(imp_score)
        
#         self.imp_score=imp_score.to(device)
        
#     def common(self, review):

#         s_e=review                                    #(4,40,768)

#         h0 = torch.zeros(2, s_e.size(0), 512 // 2)
#         c0 = torch.zeros(2, s_e.size(0), 512 // 2)
#         h0, c0 = h0.to(device), c0.to(device)
#         s_e, (hn, cn) = self.lstm_1(s_e, (h0, c0))    #(4,40,512)

#         l = self.relu(self.linear_sen1(s_e))
#         l = self.dropout1(l)
#         l = self.relu(self.linear_sen2(l))
#         l = self.dropout1(l)
#         l = self.relu(self.linear_sen3(l))     #(4,40,2)


#         comp=self.category(s_e)         #(4,40,7)
#         comp = comp.permute(0,2,1)      #(4,7,40)
        
        
#         wts = F.softmax(comp, dim=2) #(4,7,40)
#         e=torch.bmm(wts,s_e)       #(4,7,512)

#         out_sen=torch.matmul(wts,l)  #(4,7,2)

        
#         l = torch.reshape(e, (s_e.size(0), -1)) #(4,7*512)
        
#         l = self.relu(self.linear1(l))
#         l = self.dropout1(l)
#         l = self.relu(self.linear2(l))
#         l = self.dropout1(l)
#         l = self.relu(self.linear3(l))

#         model_output = self.sigmoid(self.last_dense(l))
#         model_output_sen =self.sigmoid(out_sen)

#         return model_output, e, wts,model_output_sen

    
#     def forward(self, review1,review2,review3):

#         aspect_output1,dse1,imp_s1,out_sen1=self.common(review1)
#         aspect_output2,dse2,imp_s2,out_sen2=self.common(review2)
#         aspect_output3,dse3,imp_s3,out_sen3=self.common(review3)

#         i_p_aspect = F.softmax(self.imp_score,dim=1).to(device)  #(1,7)
#         i_p_aspect= i_p_aspect.unsqueeze(dim=1) #(1,1,7)
        
#         dse1=torch.matmul(i_p_aspect,dse1).squeeze(dim=1)  #(n_p,512)
#         dse2=torch.matmul(i_p_aspect,dse2).squeeze(dim=1)  #(n_p,512)
#         dse3=torch.matmul(i_p_aspect,dse3).squeeze(dim=1)  #(n_p,512)

#         final_des_embed=torch.cat((dse1,dse2,dse3),dim=1)  #(n_p,512*3)

#         x=self.relu(self.linear_des1(final_des_embed))  #(n_p,256)
#         x=self.relu(self.linear_des2(x))  #(n_p,64)
#         x=self.relu(self.linear_des3(x)) # (n_p,8)
#         x=self.relu(self.linear_des4(x)) # (n_p,1)

#         output=self.sigmoid(x)
        
#         return aspect_output1,aspect_output2,aspect_output3,imp_s1,imp_s2,imp_s3,output,out_sen1,out_sen2,out_sen3
        
        
# text_model = BertModel(768,7)
# text_model.to(device)
# text_model.load_state_dict(torch.load('/home/sandeep_2121cs29/hardik/COLING_2022/ckpt/bert_coling_multi.pt',map_location=device))


# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('stsb-roberta-base')
# model=model.to(device)

# def create_embeddings(r):
#     sentences=nltk.sent_tokenize(r)
#     max_length=40

#     if len(sentences)<=max_length:
#         sentences=sentences+[""]*(max_length-len(sentences))

#     else:
#         sentences=sentences[0:max_length]

#     encoded=model.encode(sentences, show_progress_bar=False)
    
#     return encoded


# def f(x):
   
#   r=x.split('[DEPT]')
#   x1=r[0]
#   x2=r[1]
#   x3=r[2]

#   e1=create_embeddings(x1)
#   e2=create_embeddings(x2)
#   e3=create_embeddings(x3)

#   #   inputs = tokenizer(x,add_special_tokens=False,
#   #     return_token_type_ids=True,
#   #     return_length = True,
#   #     truncation=False)
#   #   # #inputs = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).cuda()
            
#   #   ids = torch.tensor(inputs['input_ids']).to(device,dtype = torch.long)
#   #   mask = torch.tensor(inputs['attention_mask']).to(device,dtype = torch.long)
#   #   token_type_ids = torch.tensor(inputs['token_type_ids']).to(device,dtype = torch.long)
#   #t1=(ids,mask,token_type_ids)
  
#   out1,out2,out3, attn_t1,attn_t2,attn_t3 ,d,out_sen1,out_sen2,out_sen3 = text_model(e1,e2,e3)

#   ret = d.cpu().detach().numpy()

#   print(ret)
#   return ret


# def model_prediction_gpu(x):
#     tv = torch.tensor([tokenizer.encode(v, padding='max_length', 
#                                         max_length=512, truncation=True) for v in x]).to(device,dtype = torch.long)
#     attention_mask = (tv!=0).type(torch.int64).to(device,dtype = torch.long)
#     token_type_ids = torch.zeros_like(attention_mask).to(device,dtype = torch.long)
#     t1=(tv,attention_mask,token_type_ids)
#     out_test, attn_T,attn_T_S,out_test_senti = text_model(t1,'last')
#     val = torch.logit(out_test).detach().cpu().numpy()
#     return val


method = "custom tokenizer"

# build an explainer by passing a transformers tokenizer
if method == "transformers tokenizer":
    explainer = shap.Explainer(f, tokenizer, output_names=labels)

# build an explainer by explicitly creating a masker
elif method == "default masker":
    masker = shap.maskers.Text(r"\W") # this will create a basic whitespace tokenizer
    explainer = shap.Explainer(f, masker, output_names=labels)

# build a fully custom tokenizer
elif method == "custom tokenizer":
    import re

    def custom_tokenizer(s, return_offsets_mapping=True):
        """ Custom tokenizers conform to a subset of the transformers API.
        """
        pos = 0
        offset_ranges = []
        input_ids = []
        for m in re.finditer(r"\W", s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])
        out = {}
        out["input_ids"] = input_ids
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out


x = ["this work studies the predictive uncertainty issue of deep learning models . in particular  this work focuses on the distributional uncertainty which is caused by distributional mismatch between training and test examples . the proposed method is developed based on the existing work called dirichlet prior network  dpn  . it aims to address the issue of dpn that its loss function is complicated and makes the optimization difficult . instead  this paper proposes a new loss function for dpn  which consists of the commonly used crossentropy loss term and a regularization term . two loss functions are respectively defined over indomain training examples and outofdistribution  ood  training examples . the final objective function is a weighted combination of the two loss functions . experimental study is conducted on one synthetic dataset and two image datasets  cifar10 and cifar100  to demonstrate the properties of the proposed method and compare its performance with the relevant ones in the literature . the issue researched in this work is of significance because understanding the predictive uncertainty of a deep learning model has its both theoretical and practical value . the motivation [DEPT] research issues and the proposed method are overall clearly presented . the current recommendation is weak reject because the experimental study is not convincing or comprehensive enough . 1 .although the goal of this work is to deal with the inefficiency issue of the objective function of existing dpn with the newly proposed one  this experimental study does not seem to conduct sufficient experiments to demonstrate the advantages  say  in terms of training efficiency  the capability in making the network scalable for more challenging dataset  of the proposed objective function over the existing one  2 . table 1 compares the proposed method with odin . however  as indicated in this work  odin is trained with indomain examples only . is this comparison fair  actually  odin s setting seems to be more practical and more challenging than the setting used by the propose methods . 3 .the evaluation criteria shall be better explained at the beginning of the experiment  especially how they can be collectively used to verify that the proposed method can better distinguish distributional uncertainty from other uncertainty types . 4 .in addition  the experimental study can be clearer on the training and test splits . [DEPT] how many samples from cifar10 and cifar100 are used for training and test purpose  respectively  also  since training examples are from cifar10 and cifar100 and the test examples are also from these two datasets  does this contradict with the motivation of distributional mismatch between training and test examples mentioned in the abstract  5 .the experimental study can have more comparison on challenging datasets with more classes since it is indicated that dpn has difficulty in dealing with a large number of classes . minor  1 . please define the hattheta in eq .also  is the dirac delta estimation a good enough approximation here  2 .the lambda  out   lambda  in  in eq .  11  needs to be better explained . in particular  are the first terms in eq .  10  and eq .  11  comparable in terms of magnitude  otherwise  lambda  out   lambda  in  may not make sense . 3 .the novelty and significance of finetuning the proposed model with noisy ood training images can be better justified ."]

masker = shap.maskers.Text("transformers tokenizer")
# explainer = shap.Explainer(model_prediction_gpu, tokenizer, output_names=labels)
# shap_values = explainer(x)

# shap.plots.text(shap_values)