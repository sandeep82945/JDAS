
import os
import jsonlines
import nltk
import pandas as pd 
nltk.download('punkt',quiet=True)
import sys
from tqdm import tqdm
import pickle
import json


class DataLoading():
    def __init__(self):
        super().__init__()
        
    def preprocess_text(self,string:str):
        string=string.lower()
        punctuations = '''!()-[]{};:'"\<>/?@#$^&*_~=+,"'''
        string=string.replace('�'," ")
        string=string.replace('\n',"")
        for x in string.lower(): 
            if x in punctuations: 
                string = string.replace(x, "") 

        return string
    
    def general_statistics(self,data:dict):
        
        l=0   
        count_aspects={}
        n_p=0
        
        n_p=len(data.keys())
        
        
        for k in data.keys():
            l+=len(data[k])    
            
            for v1,v2 in data[k].items():
                for v3 in v2:
                    if(v3 not in count_aspects.keys()):
                        count_aspects[v3]=0
                    else:
                        count_aspects[v3]+=1
                        
                        
        
        print(f"Total number of different papers covered: {n_p}")
        print(f"Total number of sentences captured: {l}\n")
        
        print(f"ASPECT NAME  \t COUNT\n")
        
        for k,v in count_aspects.items():
            print(f"{k}  \t {v}")
            
        
        
    
    def load_dataset(self,input_path:str):
        count_no=0
        
        
        data_path=input_path
        data_full={}
        
        
        with jsonlines.open(os.path.join(data_path,'review_with_aspect.jsonl')) as f:
            
            pbar=f.iter()
            
            for line in tqdm(pbar,desc='Loading Datasets'):
                id1=line['id']
                s=line['text']
                labels=line['labels']

                enter=nltk.sent_tokenize(s)
                enter=[self.preprocess_text(s56) for s56 in enter]
                enter=" ".join(enter)
                
                la=[]
                for k in labels:
                    if(k[2].startswith('summary')):
                        continue
                        
                    a=k[2].split('_')
                    if(a[0]=='meaningful'):
                        a=a[0]+"_"+a[2]
                    else:
                        a=a[0]+"_"+a[1]
                    
                    la.append(a)

                la=list(set(la))

                if id1 not in data_full.keys():
                    data_full[id1]={}
                    data_full[id1]["Reviews"]={}
                    
                data_full[id1]["Reviews"][enter]=la
                data_full[id1]['decision']=""
                
            
        data_path=r'data/dataset/'

        for conf in os.listdir(data_path):
            
            for dire in (os.listdir(os.path.join(data_path,conf))):
                if(dire.endswith('_content')):
                    continue
                
                if(dire.endswith('_paper')): 
                    for paper in tqdm(os.listdir(os.path.join(data_path,conf,dire)),desc=conf+" DECISIONS "+": done"):

                        with open(os.path.join(data_path,conf,dire,paper)) as out:
                            file1=json.load(out)
                        
                        if file1['id'] not in data_full.keys():
                            
                            continue
                        
                        decision=file1['decision']
                        data_full[file1['id']]['decision']='accept' if ('Accept' in decision or 'Track' in decision) else 'reject'
                        
                        
        print()
        print("Total number of papers    :",len(data_full))
        print("Number of Accepted Papers :",[k['decision'] for k in data_full.values()].count('accept'))
        print("Number of Rejected Papers :",[k['decision'] for k in data_full.values()].count('reject'))

        return data_full
                    
            
if __name__=='__main__':
    obj=DataLoading()
    input_path="data/dataset/aspect_data/"

    data_cmu=pd.DataFrame()

    data=obj.load_dataset(input_path)
    
    with open('data/created_files/dataframe_coling_reviews.json','w') as out:
        json.dump(data,out)

    
                    
        
        
            
            
            
            
            
        