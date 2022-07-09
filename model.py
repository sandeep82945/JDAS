
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from transformers import AutoModel
model_name='bert-base-uncased'

#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:2")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")



class BertModel(nn.Module):
    def __init__(self, in_features, out_features,):

        super(BertModel, self).__init__()
        self.model=AutoModel.from_pretrained(model_name,  output_hidden_states=True)

        #for param in self.model.parameters():
        #param.require_grad=True
          
        self.in_features = in_features   #768
        self.out_features = out_features    #7

        self.flatten=nn.Flatten()
        hidden_dimension=200
        self.hidden_dimension=hidden_dimension

        self.lstm_1 = nn.LSTM(in_features, hidden_dimension//2, batch_first=True, bidirectional=True) #bidirectional=True
    
        self.linear_start=nn.Linear(hidden_dimension*out_features,hidden_dimension*7)
        self.linear1=nn.Linear(hidden_dimension*7,hidden_dimension*4)
        self.linear2=nn.Linear(hidden_dimension*4,256)
        self.linear3=nn.Linear(256,64)
        self.last_dense = nn.Linear(64, self.out_features)

        self.linear1_sen=nn.Linear(hidden_dimension,64)
        self.linear2_sen=nn.Linear(64,16)
        self.linear3_sen=nn.Linear(16,2)


        self.dropout1=nn.Dropout(p=0.5)
        self.dropout2=nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        self.category=nn.Linear(hidden_dimension,out_features)


    def forward(self, t1,strategy:str):
        # print("===========================================================")
        ids, mask, token_type_ids = t1
        # print(token_type_ids)
        encoded_layers = self.model(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)[2]
        # print("=================================================================")
        scibert_hidden_layer = encoded_layers
        # exit(0)
        if(strategy=='last_4'):
          scibert_hidden_layers=torch.cat((scibert_hidden_layer[-1],
                                        scibert_hidden_layer[-2],
                                        scibert_hidden_layer[-3],
                                        scibert_hidden_layer[-4]),dim=2)
          
        if(strategy=='last'):
          scibert_hidden_layers=encoded_layers[12]


        if(strategy=='mean'):
          scibert_hidden_layers=torch.mean(encoded_layers,dim=2)
      

        s_e=scibert_hidden_layers                                  #(8,500,768)

        h0 = torch.zeros(2, s_e.size(0),  self.hidden_dimension // 2)
        c0 = torch.zeros(2, s_e.size(0),  self.hidden_dimension // 2)

        h0, c0 = h0.to(device), c0.to(device)
        s_e, (hn, cn) = self.lstm_1(s_e, (h0, c0))                   #(8,500,200)

        
       #ASECT EMBEDDING COMPUTATION
        comp = self.tanh(self.category(s_e))                         #(8,500,7)
        comp = comp.permute(0,2,1)                                   #(8,7,500)
        wts = F.softmax(comp, dim=2)                                 #(8,7,500)
        e   = torch.bmm(wts,s_e)                                     #(8,7,200)

        #SENTIMENT EMBEDDINGS COMPUTATION
        comp_sent=  self.relu(self.linear1_sen(s_e))                      #(8,500,64)
        comp_sent = self.relu(self.linear2_sen(comp_sent))                #(8,500,16)
        comp_sent = self.relu(self.linear3_sen(comp_sent))                #(8,500,2)
        comp_sent=torch.bmm(wts,comp_sent)                                #(8,7,2) 
        
        l = torch.reshape(e, (ids.size(0), -1))        #(8,14*200)
        l=  self.relu(self.linear_start(l))                #(8,7*200)
        l = self.relu(self.linear1(l))                     #(8,2*200)
        #l = self.dropout1(l)
        l = self.relu(self.linear2(l))                     #(8,256)
        #l = self.dropout1(l)
        l = self.relu(self.linear3(l))                     #(8,64)

        model_output = self.sigmoid(self.last_dense(l))    #(8,14)
        model_output_sent = self.sigmoid(comp_sent)             #(8,40,7,2)
        
        del l,comp,s_e,hn,cn,scibert_hidden_layer,ids,mask,token_type_ids
      
        return model_output, wts,model_output_sent
