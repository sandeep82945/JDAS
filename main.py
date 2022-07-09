
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_finetuning import DecsionModel
import torch
import pickle

from tqdm.autonotebook import tqdm
from dataloader import getLoaders
import matplotlib.pyplot as plt

trainloader, validloader, testloader = getLoaders(batch_size=4)

print("Length of TrainLoader:",len(trainloader))
print("Length of ValidLoader:",len(validloader))
print("Length of TestLoader:",len(testloader))

#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:2")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

from model_finetuning import BertModel
aspect_model = BertModel(768,7)
model_path="/home/sandeep_2121cs29/hardik/COLING_2022/ckpt/aspect_sentiment_model.pt"
        
aspect_model.load_state_dict(torch.load(model_path,map_location=device))
aspect_model=aspect_model.train()

text_model = DecsionModel(aspect_model)
text_model.to(device)
criterion1 = nn.BCELoss()
criterion2 = nn.BCELoss()
criterion3 = nn.BCELoss()

from transformers import AdamW, get_linear_schedule_with_warmup

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in text_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in text_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
]
# optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0.001, weight_decay=1e-4)

# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)


text_model.train()
result=[]
EPOCH=10

train_out = []
val_out = []
train_true = []
val_true = []
test_out = []
test_true = []
attn_train = []
attn_val = []
attn_test = []
attn_test_senti=[]
test_out_senti=[]
test_true_senti=[]
loss_log1 = []
loss_log2 = []


for epoch in range(EPOCH):

  final_train_loss=0.0
  final_val_loss=0.0
  l1 = []
  text_model.train()

  for data in tqdm(trainloader,desc="Train epoch {}/{}".format(epoch + 1, EPOCH)):

    ids1 = data['ids1'].to(device,dtype = torch.long)
    mask1 = data['mask1'].to(device,dtype = torch.long)
    token_type_ids1 = data['token_type_ids1'].to(device,dtype = torch.long)
    targets1 = data['targets1'].to(device,dtype = torch.float)
    targets_s1 = data['targets_senti1'].to(device,dtype = torch.float)

    ids2 = data['ids2'].to(device,dtype = torch.long)
    mask2 = data['mask2'].to(device,dtype = torch.long)
    token_type_ids2 = data['token_type_ids2'].to(device,dtype = torch.long)
    targets2 = data['targets2'].to(device,dtype = torch.float)
    targets_s2 = data['targets_senti2'].to(device,dtype = torch.float)

    ids3 = data['ids3'].to(device,dtype = torch.long)
    mask3 = data['mask3'].to(device,dtype = torch.long)
    token_type_ids3 = data['token_type_ids3'].to(device,dtype = torch.long)
    targets3 = data['targets3'].to(device,dtype = torch.float)
    targets_s3 = data['targets_senti3'].to(device,dtype = torch.float)
      
    des=data['decision'].to(device,dtype = torch.float)

    t1 = (ids1,mask1,token_type_ids1)
    t2 = (ids2,mask2,token_type_ids2)
    t3 = (ids3,mask3,token_type_ids3)
    
    optimizer.zero_grad()
    
    out1,out2,out3 , attn_t1,attn_t2,attn_t3, out_senti1,out_senti2,out_senti3 ,d = text_model(t1,t2,t3,'last')
    # print(out1)
    # print(targets1)
    # out2, attn_t2,out_senti2,d = text_model(t2,'last')
    # out3, attn_t3,out_senti3,d = text_model(t3,'last')
    
    
    
    loss = (criterion1(out1, targets1) + criterion1(out2, targets2)+criterion1(out3, targets3) + criterion2(out_senti1, targets_s1)+criterion2(out_senti2, targets_s2)+criterion2(out_senti3, targets_s3) )/6
    
    loss=(loss+criterion3(d,des))/2

    l1.append(loss.item())
    final_train_loss +=loss.item()
    loss.backward()
    optimizer.step()
  
    if (epoch+1 == EPOCH):
      train_out.append((torch.transpose(out1,0,1)).detach().cpu())
      train_true.append((torch.transpose(targets1,0,1)).detach().cpu())
            
      train_out.append((torch.transpose(out2,0,1)).detach().cpu())
      train_true.append((torch.transpose(targets2,0,1)).detach().cpu())

      train_out.append((torch.transpose(out3,0,1)).detach().cpu())
      train_true.append((torch.transpose(targets3,0,1)).detach().cpu())

  loss_log1.append(np.average(l1))

  text_model.eval()
  l2 = []

  for data in tqdm(validloader,desc="Valid epoch {}/{}".format(epoch + 1, EPOCH)):

    ids1 = data['ids1'].to(device,dtype = torch.long)
    mask1 = data['mask1'].to(device,dtype = torch.long)
    token_type_ids1 = data['token_type_ids1'].to(device,dtype = torch.long)
    targets1 = data['targets1'].to(device,dtype = torch.float)
    targets_s1 = data['targets_senti1'].to(device,dtype = torch.float)

    ids2 = data['ids2'].to(device,dtype = torch.long)
    mask2 = data['mask2'].to(device,dtype = torch.long)
    token_type_ids2 = data['token_type_ids2'].to(device,dtype = torch.long)
    targets2 = data['targets2'].to(device,dtype = torch.float)
    targets_s2 = data['targets_senti2'].to(device,dtype = torch.float)

    ids3 = data['ids3'].to(device,dtype = torch.long)
    mask3 = data['mask3'].to(device,dtype = torch.long)
    token_type_ids3 = data['token_type_ids3'].to(device,dtype = torch.long)
    targets3 = data['targets3'].to(device,dtype = torch.float)
    targets_s3 = data['targets_senti3'].to(device,dtype = torch.float)
       
    t1 = (ids1,mask1,token_type_ids1)
    t2 = (ids2,mask2,token_type_ids2)
    t3 = (ids3,mask3,token_type_ids3)
    
    optimizer.zero_grad()
    
    out1,out2,out3 , attn_t1,attn_t2,attn_t3, out_senti1,out_senti2,out_senti3 ,d = text_model(t1,t2,t3,'last')
    # out2, attn_t2,out_senti2 = text_model(t2,'last')
    # out3, attn_t3,out_senti3 = text_model(t3,'last')
    
    loss = (criterion1(out1, targets1) + criterion1(out2, targets2)+criterion1(out3, targets3) + criterion2(out_senti1, targets_s1)+criterion2(out_senti2, targets_s2)+criterion2(out_senti3, targets_s3) )/6
    
    l2.append(loss.item())
    final_val_loss+=loss.item()

    if (epoch+1 == EPOCH):
     
      val_out.append((torch.transpose(out1,0,1)).detach().cpu())
      val_true.append((torch.transpose(targets1,0,1)).detach().cpu())

      val_out.append((torch.transpose(out2,0,1)).detach().cpu())
      val_true.append((torch.transpose(targets2,0,1)).detach().cpu())

      val_out.append((torch.transpose(out3,0,1)).detach().cpu())
      val_true.append((torch.transpose(targets3,0,1)).detach().cpu())

    
  loss_log2.append(np.average(l2))
  curr_lr = optimizer.param_groups[0]['lr']


  print("Epoch {}, loss: {}, val_loss: {}".format(epoch+1, final_train_loss/len(trainloader), final_val_loss/len(validloader)))
  print()
  

with torch.no_grad():
   for data in tqdm(testloader,desc="TEST epoch {}/{}".format(epoch + 1, EPOCH)):
    
    ids1 = data['ids1'].to(device,dtype = torch.long)
    mask1 = data['mask1'].to(device,dtype = torch.long)
    token_type_ids1 = data['token_type_ids1'].to(device,dtype = torch.long)
    targets1 = data['targets1'].to(device,dtype = torch.float)

    ids2 = data['ids2'].to(device,dtype = torch.long)
    mask2 = data['mask2'].to(device,dtype = torch.long)
    token_type_ids2 = data['token_type_ids2'].to(device,dtype = torch.long)
    targets2 = data['targets2'].to(device,dtype = torch.float)

    ids3 = data['ids3'].to(device,dtype = torch.long)
    mask3 = data['mask3'].to(device,dtype = torch.long)
    token_type_ids3 = data['token_type_ids3'].to(device,dtype = torch.long)
    targets3 = data['targets3'].to(device,dtype = torch.float)

       
    t1 = (ids1,mask1,token_type_ids1)
    t2 = (ids2,mask2,token_type_ids2)
    t3 = (ids3,mask3,token_type_ids3)
    
    optimizer.zero_grad()
    
    out_test1,out_test2,out_test3 , attn_t1,attn_t2,attn_t3, out_senti1,out_senti2,out_senti3 ,d = text_model(t1,t2,t3,'last')
    # out_test2, attn_t2 = text_model(t2,'last')
    # out_test3, attn_t3 = text_model(t3,'last')
    
   
    test_out.append((torch.transpose(out_test1,0,1)).detach().cpu())
    test_true.append((torch.transpose(targets1,0,1)).detach().cpu())
    attn_test.append((torch.tensor(attn_t1.clone().detach().cpu())))

    test_out.append((torch.transpose(out_test2,0,1)).detach().cpu())
    test_true.append((torch.transpose(targets2,0,1)).detach().cpu())
    attn_test.append((torch.tensor(attn_t2)).detach().cpu())
    
    test_out.append((torch.transpose(out_test3,0,1)).detach().cpu())
    test_true.append((torch.transpose(targets3,0,1)).detach().cpu())
    attn_test.append((torch.tensor(attn_t3)).detach().cpu())
    

plt.plot(range(len(loss_log1)), loss_log1)
plt.plot(range(len(loss_log2)), loss_log2)
plt.savefig('graphs/loss_bert_multi.png')

torch.save(text_model.state_dict(), "ckpt/bert_coling_multi.pt")

train_out = torch.cat(train_out, 1)
train_true = torch.cat(train_true, 1)

val_out = torch.cat(val_out, 1)
val_true = torch.cat(val_true, 1)

# print(val_out.size())
# print(val_true.size())
# print("====================================================")

test_out = torch.cat(test_out, 1)
test_true = torch.cat(test_true, 1)
attn_test = torch.cat(attn_test, 0)

train_out, val_out, train_true, val_true = train_out.cpu(), val_out.cpu(), train_true.cpu(), val_true.cpu()
test_out, test_true = test_out.cpu(), test_true.cpu()
attn_test = attn_test.cpu()

attnfile = open('outputs/attn_noaspect_multi.pkl', 'wb')
pickle.dump(attn_test, attnfile)


test_out_ = (test_out, test_true)

test_outs = open('outputs/main_bert7_test_out_noaspect_multi.pkl', 'wb')
pickle.dump(test_out_, test_outs)


f=open("results/"+"bert_multi"+".txt",'w')
f.close()

def labelwise_metrics(pred, true, split):
  f=open("results/"+"bert_multi"+".txt",'a')
  f.write('-'*25 + split + '-'*25 + '\n\n')
   
  # print(pred.shape)
  # print(true.shape)

  pred = (pred>0.425)

  batch_size = len(pred)
  
  pred = pred.to(torch.int)
  true = true.to(torch.int)

  from sklearn.metrics import accuracy_score
  from sklearn.metrics import confusion_matrix

  for i in range(batch_size):
    acc=accuracy_score(true[i],pred[i])

    epsilon = 1e-7
    confusion_vector = pred[i]/true[i]

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    precision = true_positives/(true_positives+false_positives+epsilon)
    recall = true_positives/(true_positives+false_negatives+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)

    print("Label: {}, acc: {:.3f}, f1: {:.3f}".format(i+1, acc, f1))
    f.write("Label: {}, acc: {:.3f}, f1: {:.3f}\n".format(i+1, acc, f1))
    f.write(str(confusion_matrix(true[i], pred[i])))
    f.write('\n')

  return 0

f1=open("results/"+"bert_multi_senti"+".txt",'w')
f1.close()

print('Training...')
labelwise_metrics(train_out, train_true, 'TRAINING')
print()
print('Validation...')
labelwise_metrics(val_out, val_true, 'VALIDATION')
print()
print('Test...')
labelwise_metrics(test_out, test_true, 'TESTING')