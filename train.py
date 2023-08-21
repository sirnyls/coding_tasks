import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch
from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import set_seed
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,classification_report
from datasets import load_metric
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch import nn
import transformers
import wandb

from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    trainer.optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
)
trainer.optimizer.set_scheduler(scheduler)

def compute_metrics_discrete(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    cr=classification_report(labels,predictions,output_dict=True)
    recall_w = recall_score(y_true=labels, y_pred=predictions,average='weighted')
    precision_w = precision_score(y_true=labels, y_pred=predictions,average='weighted')
    f1_micro = f1_score(y_true=labels, y_pred=predictions,average='micro')
    f1_weighted = f1_score(y_true=labels, y_pred=predictions,average='weighted')
    return {"accuracy": accuracy, "f1_0":cr['0']['f1-score'],"f1_1":cr['1']['f1-score'],
            "precision_1":cr['1']['precision'],"recall_1":cr['1']['recall'],
             "precision_w": precision_w, "recall_w": recall_w,
            "f1_micro": f1_micro,"f1_weighted": f1_weighted} 

def process_data(file_path,dataset,amr=True,outcome_variable='helpfulness'):
    """Process data for training RoBERTa model, formatting depends on the dataset"""
    df=pd.read_csv(file_path)
    if amr:
        if dataset in ['PAWS']:
            df=df.assign(text="Sentence 1: "+df.premise_+"\nAMR 1: "+df.amr_p+"\nSentence 2: "+df.hypothesis_+"\nAMR 2: "+df.amr_h)
        elif dataset in ['translation','logic','django','spider']:
            df=df.assign(text="Text: "+df.text+"\nAMR: "+df.amr)
        elif dataset in ['pubmed']:
            df=df.assign(text="Text: "+df.text+"\nInteraction: "+df.interaction+"\nAMR: "+df.amr)
    else:
        if dataset in ['PAWS']:
            df=df.assign(text="Sentence 1: "+df.amr_p+"\nSentence 2: "+df.hypothesis_)
        elif dataset in ['translation','logic','django','spider']:
            df=df.assign(text="Text: "+df.text)
        elif dataset in ['pubmed']:
            df=df.assign(text="Text: "+df.text+"\nInteraction: "+df.interaction)
    
    if outcome_variable=='helpfulness':
        df=df.assign(label=np.where(df.helpfulness<=0,0,1))
    elif outcome_variable=='did_llm_failed':
        df=df.assign(label=df.did_llm_failed)
    df=df.loc[:,['id','text','label']]
    df=df.loc[~df.text.isna()]
    print df
    return df


def split_sets(dataset,df):
    """Split data into train, dev and test sets, formatting depends on the dataset"""
    if dataset in ['translation']:
        df['set']=df.id.str[:10]
        train_set=df.loc[df['set']=='newstest13']
        dev_set, test_set = train_test_split(df.loc[df['set']=='newstest16'], test_size=0.5,random_state=42)
    elif dataset in ['PAWS','pubmed']:
        train_set, val_df = train_test_split(df, test_size=0.3,random_state=42)
        dev_set, test_set = train_test_split(val_df, test_size=0.5,random_state=42)
    elif dataset in ['logic','django','spider']:
        train_set=df.loc[df['id'].str.contains('train')]
        test_set=df.loc[df['id'].str.contains('test')]
        dev_set=df.loc[df['id'].str.contains('dev')]
    
    return train_set,dev_set,test_set

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

def model_init():
    transformers.set_seed(42)
    m = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2,device_map='auto')
    m.roberta.apply(freeze_weights)
    for name, param in m.classifier.named_parameters():
        param.requires_grad = True
    return m