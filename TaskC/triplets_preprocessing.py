#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import argparse

# In[2]:


# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("input_info", help = "Path to Input File in CSV Format")
parser.add_argument("input_predicate", help = "Path to Input File in CSV Format")
parser.add_argument("output", help = "Path to Output File in CSV Format")
parser.add_argument("output_a", help = "Path to Output File in CSV Format")
parser.add_argument("output_b", help = "Path to Output File in CSV Format")
parser.add_argument("output_c", help = "Path to Output File in CSV Format")
parser.add_argument("output_d", help = "Path to Output File in CSV Format")
 
# Read arguments from command line
args = parser.parse_args()
 
input_info = args.input_info
input_predicate = args.input_predicate
output_file = args.output
output_a = args.output_a
output_b = args.output_b
output_c = args.output_c
output_d = args.output_d


df_info = pd.read_csv(input_info)
df = pd.read_csv(input_predicate)


# In[3]:


df_info = df_info[['topic', 'paper_ID', 'pos1', 'labels']]
df_info.columns = ['topic', 'paper_ID', 'sentence_ID', 'info-unit']


# In[4]:


df.drop(columns = ['input'], inplace = True)


# In[5]:


df = df.astype({'labels' : 'int64'})


# In[6]:


df


# In[7]:


df = pd.merge(df_info, df, on = ['topic', 'paper_ID', 'sentence_ID'])


# In[8]:


df


# In[9]:

num = []
pre, paper, sent_id, n = "", 0, 0, 0
for i in range(len(df)):
    cur = df.iloc[i, 0]
    pid = df.iloc[i, 1]
    sid = df.iloc[i, 2]
    if (cur == pre and pid == paper and sid == sent_id):
        n += 1
    else:
        n = 0
    num.append(n)
    pre, paper, sent_id = cur, pid, sid
df['num'] = num

# In[10]:


df.to_csv(output_file, index = None)


# In[11]:


df = df[~(df['info-unit'].isin(['research-problem', 'code']))]


# In[12]:


df_sub = df.rename(columns = {'start_index': 'start_sub', 'end_index': 'end_sub', 'phrases': 'sub', 'labels': 'labels_sub', 'num': 'sub_num'})
df_sub = df_sub[df_sub['labels_sub'] == 0]


# In[13]:


df_pred = df.rename(columns = {'start_index': 'start_pred', 'end_index': 'end_pred', 'phrases': 'pred', 'labels': 'labels_pred', 'num': 'pred_num'})
df_pred = df_pred[df_pred['labels_pred'] == 1]


# In[14]:


df_obj = df.rename(columns = {'start_index': 'start_obj', 'end_index': 'end_obj', 'phrases': 'obj', 'labels': 'labels_obj', 'num': 'obj_num'})
df_obj = df_obj[df_obj['labels_obj'] == 0]


# # Triplets_A

# In[15]:


df = pd.merge(df_sub, df_pred, on = ['topic', 'paper_ID', 'sentence_ID', 'info-unit', 'sentence'])
df = pd.merge(df, df_obj, on = ['topic', 'paper_ID', 'sentence_ID', 'info-unit', 'sentence'])
df = df[(df['sub_num'] < df['pred_num']) & (df['pred_num'] < df['obj_num'])]


# In[16]:


df.reset_index(inplace = True, drop = True)


# In[17]:


df


# In[18]:


input_sent = []
for i in range(len(df)):
    triplets = []
    triplets.append([df.loc[i, 'start_sub'], df.loc[i, 'end_sub'], df.loc[i, 'labels_sub']])
    triplets.append([df.loc[i, 'start_pred'], df.loc[i, 'end_pred'], df.loc[i, 'labels_pred']])
    triplets.append([df.loc[i, 'start_obj'], df.loc[i, 'end_obj'], df.loc[i, 'labels_obj']])
    triplets = sorted(triplets)
    sent = df.loc[i, 'sentence']
    for j, triplet in enumerate(triplets):
        if(triplet[2] == 0):
            ts, te = "[[ ", " ]]"
        else:
            ts, te = "<< ", " >>"
        x = triplet[0] + 6*j
        sent = sent[:x] + ts + sent[x:]
        x = triplet[1] + 6*j + 3
        while(x<len(sent) and sent[x]!=' '):
            x += 1
        sent = sent[:x] + te + sent[x:]
    input_sent.append(sent)
df['input'] = input_sent


# In[19]:


df.to_csv(output_a, index = None)


# # Triplets_B

# In[20]:


df = pd.merge(df_sub, df_obj, on = ['topic', 'paper_ID', 'sentence_ID', 'info-unit', 'sentence'])
df = df[(df['sub_num'] < df['obj_num'])]


# In[21]:


df.reset_index(inplace = True, drop = True)


# In[22]:


df


# In[23]:


input_sent = []
for i in range(len(df)):
    triplets = []
    triplets.append([df.loc[i, 'start_sub'], df.loc[i, 'end_sub']])
    triplets.append([df.loc[i, 'start_obj'], df.loc[i, 'end_obj']])
    sent = df.loc[i, 'sentence']
    for j, triplet in enumerate(triplets):
        if(j == 1):
            ts, te = "[[ ", " ]]"
        else:
            ts, te = "<< ", " >>"
        x = triplet[0] + 6*j
        sent = sent[:x] + ts + sent[x:]
        x = triplet[1] + 6*j + 3
        while(x<len(sent) and sent[x]!=' '):
            x += 1
        sent = sent[:x] + te + sent[x:]
    input_sent.append(sent)
df['input'] = input_sent


# In[24]:


df.to_csv(output_b, index = None)


# # Triplets_C

# In[25]:


df = pd.read_csv(output_file)


# In[26]:


df = df[~(df['info-unit'].isin(['research-problem', 'code']))]


# In[27]:


def type_C(df, label):
    df = df[df['labels'] == label].reset_index(drop = True)
    num = []
    pre, paper, sent_id, n = "", 0, 0, 0
    for i in range(len(df)):
        cur = df.iloc[i, 0]
        pid = df.iloc[i, 1]
        sid = df.iloc[i, 2]
        if (cur == pre and pid == paper and sid == sent_id):
            n += 1
        else:
            n = 0
        num.append(n)
        pre, paper, sent_id = cur, pid, sid
    df['num'] = num
    return df;


# In[28]:


df_pred = type_C(df, 1)
df_pred.rename(columns = {'start_index': 'start_pred', 'end_index': 'end_pred', 'phrases': 'pred', 'labels': 'labels_pred', 'num': 'pred_num'}, inplace = True)


# In[29]:


df_obj = type_C(df, 0)
df_obj.rename(columns = {'start_index': 'start_obj', 'end_index': 'end_obj', 'phrases': 'obj', 'labels': 'labels_obj', 'num': 'obj_num'}, inplace = True)


# In[30]:


df = pd.merge(df_pred, df_obj, on = ['topic', 'paper_ID', 'sentence_ID', 'info-unit', 'sentence'])
df = df[(df['end_pred'] < df['start_obj'])]
df.reset_index(inplace = True, drop = True)


# In[31]:


df


# In[32]:


input_sent = []
subject = []
for i in range(len(df)):
    triplets = []
    triplets.append([df.loc[i, 'start_pred'], df.loc[i, 'end_pred'], df.loc[i, 'labels_pred']])
    triplets.append([df.loc[i, 'start_obj'], df.loc[i, 'end_obj'], df.loc[i, 'labels_obj']])
    triplets = sorted(triplets)
    sent = df.loc[i, 'sentence']
    info = df.loc[i, 'info-unit']
    info = info.replace('-', ' ')
    for j, triplet in enumerate(triplets):
        if(triplet[2] == 0):
            ts, te = "[[ ", " ]]"
        else:
            ts, te = "<< ", " >>"
        x = triplet[0] + 6*j
        sent = sent[:x] + ts + sent[x:]
        x = triplet[1] + 6*j + 3
        while(x<len(sent) and sent[x]!=' '):
            x += 1
        sent = sent[:x] + te + sent[x:]
    sent = "[[ " + info + " ]] : " + sent
    input_sent.append(sent)
    subject.append(info)
df['sub'] = subject
df['input'] = input_sent


# In[33]:


df.to_csv(output_c, index = None)


# # Triplets_D

# In[34]:


df = df_obj


# In[35]:


df


# In[36]:


input_sent = []
subject = []
for i in range(len(df)):
    x = df.loc[i, 'start_obj']
    y = df.loc[i, 'end_obj'] + 3
    sent = df.loc[i, 'sentence']
    info = df.loc[i, 'info-unit']
    info = info.replace('-', ' ')
    x = df.loc[i, 'start_obj']
    sent = sent[:x] + "[[ " + sent[x:]
    x = df.loc[i, 'end_obj'] + 3
    while(x<len(sent) and sent[x]!=' '):
        x += 1
    sent = sent[:x] + " ]]" + sent[x:]
    sent = "[[ " + info + " ]] : " + sent
    input_sent.append(sent)
    subject.append(info)
df['sub'] = subject
df['input'] = input_sent


# In[37]:


df


# In[38]:


df.to_csv(output_d, index = None)

