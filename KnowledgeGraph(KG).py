#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
os.chdir("C:\\Users\\Thispc\\Downloads")


# In[2]:


import re
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm


# In[3]:


f = open('covid.txt','r')
lines=f.readlines()
texts=lines[0]


# In[4]:


with open ("cv20.csv", "r", encoding="utf-8") as myfile:
    text1=myfile.readlines()


# In[5]:


def get_entities(sent):
    ent1 = ""
    ent2 = ""
    prv_tok_dep = ""  
    prv_tok_text = "" 
    prefix = ""
    modifier = ""
    for tok in nlp(sent):
        if tok.dep_ != "punct":
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    return [ent1.strip(), ent2.strip()]


# In[6]:


text1


# In[7]:


get_entities(texts)


# In[8]:


entity_pairs = []

for i in tqdm(text1):
    entity_pairs.append(get_entities(i))


# In[9]:


entity_pairs[10:20]


# In[10]:


def get_relation(sent):

    doc = nlp(sent)
    matcher = Matcher(nlp.vocab)
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", [pattern]) 

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)


# In[11]:


get_relation(texts)


# In[12]:


relations = [get_relation(i) for i in tqdm(text1)]


# In[13]:


pd.Series(relations).value_counts()[:50]


# In[14]:


source = [i[0] for i in entity_pairs]
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
kg_df


# In[18]:


kg_df['edge'].value_counts()


# In[15]:


G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())


# In[16]:


plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='red', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[17]:


G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="CONCLUSIONS"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='red', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[19]:


G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="is available"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='red', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[20]:


G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="suggest"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='red', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[ ]:




