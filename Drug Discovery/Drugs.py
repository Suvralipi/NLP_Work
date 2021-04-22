#!/usr/bin/env python
# coding: utf-8

# In[16]:


import spacy
from spacy import displacy
import en_ner_bc5cdr_md
import sys
import os
import gc
from spacy.matcher import Matcher
from tqdm import tqdm
from spacy.util import minibatch, compounding


# In[7]:


# sets the model, output directory and training iterations 
model = en_ner_bc5cdr_md
#output_dir=Path("C:\\Users\\")
n_iter=100


# In[8]:


if model is not None:
    nlp = spacy.load('en_ner_bc5cdr_md')  # load existing spaCy model
    print("Loaded model '%s'" % model)
else:
# this will create a blank english model
    nlp = spacy.blank('en')  # create blank Language class
    print("Created blank 'en' model")


# In[9]:


LABEL='ADRS'
TRAIN_DATA = [
    (
        "arrhythmia",
        {"entities": [(0, 10, LABEL)]},
    ),
    
    (
        "hallucinations",
        {"entities": [(0, 14, LABEL)]},
    ),
    ("amnesia", {"entities": [(0, 7, LABEL)]}),
    (
        "delirium",
        {"entities": [(0,8, LABEL)]},
    ),
    ("psychosis", {"entities": [(0, 9, LABEL)]}),
]


# In[17]:


# add labels, Trains data based on annotations 
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
else:
    ner = nlp.get_pipe("ner")

ner.add_label(LABEL)  # add new entity label to entity recognizer
# Adding extraneous labels shouldn't mess anything up
#ner.add_label("VEGETABLE")
if model is None:
    optimizer = nlp.begin_training()
else:
    optimizer = nlp.entity.create_optimizer()
#move_names = list(ner.move_names)
# get names of other pipes to disable them during training
#pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
#    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches=minibatch(TRAIN_DATA,size=compounding(4., 32., 1.001))
        for batch in batches:
            texts,annotations = zip(*batch)
            # Updating the weights
            nlp.update(texts, annotations, sgd=optimizer,drop=0.35, losses=losses)
        print('Losses', losses)


# In[11]:


import random


# In[18]:


test_text = "Exposure to chloroquine was associated with a statistically significant high reporting of amnesia , delirium , hallucinations"
doc = ner_model(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)


# In[20]:


colors = {
    'CHEMICAL': 'lightpink',
    'ADRS': 'lightgreen',
}
displacy.render(doc, style='ent', options={
    'colors': colors
})


# In[ ]:




