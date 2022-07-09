
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
    def __init__(self, in_features, out_features):

        super(BertModel, self).__init__()
        self.model=AutoModel.from_pretrained('bert-base-uncased',  output_hidden_states=True,local_files_only=True)

        self.in_features = in_features   #768
        self.out_features = out_features    #7

        self.flatten=nn.Flatten()
        self.lstm_1 = nn.LSTM(in_features, 200//2, batch_first=True, bidirectional=True) #bidirectional=True
    
      
        self.linear1=nn.Linear(200*out_features,200*2)
        self.linear2=nn.Linear(200*2,256)
        self.linear3=nn.Linear(256,64)

        self.linear1_sen=nn.Linear(200,64)
        self.linear2_sen=nn.Linear(64,2)
   
        self.last_dense = nn.Linear(64, self.out_features)
        self.dropout1=nn.Dropout(p=0.5)
        self.dropout2=nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        category = torch.rand(200, out_features,requires_grad=True)  #(512,7)
        nn.init.xavier_normal_(category)
       
        self.category=category.to(device)
       
    def forward(self, t1,strategy:str):
        
        ids, mask, token_type_ids = t1
        encoded_layers = self.model(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)[2]
        
        scibert_hidden_layer = encoded_layers
        
        if(strategy=='last_4'):
          scibert_hidden_layers=torch.cat((scibert_hidden_layer[-1],
                                        scibert_hidden_layer[-2],
                                        scibert_hidden_layer[-3],
                                        scibert_hidden_layer[-4]),dim=2)
          
        if(strategy=='last'):
          scibert_hidden_layers=encoded_layers[12]


        if(strategy=='mean'):
          scibert_hidden_layers=torch.mean(encoded_layers,dim=2)
      

        s_e=scibert_hidden_layers                  #(32,13,768)

        h0 = torch.zeros(2, s_e.size(0), 200 // 2)
        c0 = torch.zeros(2, s_e.size(0), 200 // 2)
        h0, c0 = h0.to(device), c0.to(device)
        s_e, (hn, cn) = self.lstm_1(s_e, (h0, c0))    #(32,50,200)
      
  
        c=self.category.unsqueeze(0)                       #(1,512,7)
        comp = torch.matmul(s_e,c)                         #(32,13,7)
        comp = comp.permute(0,2,1)                         #(32,7,13)

        comp1=    self.relu(self.linear1_sen(s_e))         #(32,50,256)
        comp1 =   self.linear2_sen(comp1)                  #(32,50,2)
        
        wts = F.softmax(comp, dim=2) #(32,7,50)
        wts1= torch.bmm(wts,comp1)   #(32,7,2)
      
        e=torch.bmm(wts,s_e)         #(32,7,200)

        l = torch.reshape(e, (ids.size(0), 200*7))

        l = self.relu(self.linear1(l))
        l = self.dropout1(l)
        l = self.relu(self.linear2(l))
        l = self.dropout1(l)
        l = self.relu(self.linear3(l))

        model_output = self.sigmoid(self.last_dense(l))
        model_output_sent = self.sigmoid(wts1)
        
        del l,comp,s_e,hn,cn,scibert_hidden_layer,ids,mask,token_type_ids
      
        return model_output,wts,model_output_sent,e




class DecsionModel(nn.Module):
    def __init__(self,aspect_model):

        super(DecsionModel, self).__init__()
        
        model_path="/home/sandeep_2121cs29/hardik/COLING_2022/ckpt/aspect_sentiment_model.pt"
        
        self.model=aspect_model
        #self.model.train()

        hidden_dimension=200
    
        self.linear1=nn.Linear(hidden_dimension*3,256)
        self.linear2=nn.Linear(256,64)
        self.linear3=nn.Linear(64,8)
        self.linear4=nn.Linear(8,1)


        self.dropout1=nn.Dropout(p=0.5)
        self.dropout2=nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        
        imp_score= torch.rand( 1,7,requires_grad=True)  #(512,1)
        nn.init.xavier_normal_(imp_score)
        
        self.imp_score=imp_score.to(device)
        

    def forward(self, t1,t2,t3,s):
      
        o1,a_w1,os1,ase_1=self.model(t1,s)        
        o2,a_w2,os2,ase_2=self.model(t2,s)
        o3,a_w3,os3,ase_3=self.model(t3,s)

        #aspect_weights(a_w1)-(n_p,7,500)
        #aspect_specific_embeddings(ase_1)= (n_p,7,200)
       
        i_p_aspect = F.softmax(self.imp_score,dim=1).to(device)  #(1,7)
        i_p_aspect= i_p_aspect.unsqueeze(dim=1) #(1,1,7)
        
        dse1=torch.matmul(i_p_aspect,ase_1).squeeze(dim=1)  #(n_p,200)
        dse2=torch.matmul(i_p_aspect,ase_2).squeeze(dim=1)  #(n_p,200)
        dse3=torch.matmul(i_p_aspect,ase_3).squeeze(dim=1)  #(n_p,200)

        final_des_embed=torch.cat((dse1,dse2,dse3),dim=1)  #(n_p,200*3)

        x=self.relu(self.linear1(final_des_embed))  #(n_p,256)
        x=self.relu(self.linear2(x))  #(n_p,64)
        x=self.relu(self.linear3(x)) # (n_p,8)
        x=self.relu(self.linear4(x)) # (n_p,1)

        output=self.sigmoid(x)

        return  o1,o2,o3,a_w1,a_w2,a_w3,os1,os2,os3,output
        






        





        

