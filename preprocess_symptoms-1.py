#!/usr/bin/env python
# coding: utf-8

# In[4]:

from numba import jit, cuda
import pandas as pd
import gzip as gzip

# df = pd.read_csv("NOTEEVENTS.csv", nrows=20)
csvFileNoteEvents = gzip.open("NOTEEVENTS.csv.gz",'rb')
# print(csvFileNoteEvents)
@jit(target ="cuda")
def process(csvFileNoteEvents):
    df = pd.read_csv(csvFileNoteEvents,low_memory = False)
    # print(df)


    # In[5]:


    # df1 = df[df["HADM_ID"]==142582]
    # df1.head()
    # df1.to_csv("temp_NOTEEVENTS2.csv")
    df_samples = df.sample(n = 5)


    # In[6]:


    df_events = df_samples[['HADM_ID', 'TEXT']]


    # In[7]:


    csvFileAdmissions = gzip.open("ADMISSIONS.csv.gz",'rb')
    df_adm = pd.read_csv(csvFileAdmissions)
    df_admissions = df_adm[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'DIAGNOSIS']]


    # In[8]:


    df_merge = pd.merge(df_events, df_admissions, on='HADM_ID', how='inner')


    # In[1]:


    from keybert import KeyBERT
    from flair.embeddings import TransformerDocumentEmbeddings

    roberta = TransformerDocumentEmbeddings('PlanTL-GOB-ES/roberta-base-biomedical-clinical-es')
    kw_model = KeyBERT(model=roberta)


# In[ ]:


    import re
    def process_symptoms(text):
        text = re.sub(r"\[.*\]", " ", text)
        text = re.sub(r"_{2,}", " ", text)
        text = re.sub(r"\n", " ", text)
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(4, 4), use_mmr=True, diversity=0.5, top_n=20)
        symptoms = [x[0] for x in keywords]
        symptoms_str = ",".join(symptoms)
        print(".")
        return symptoms_str


    # In[ ]:


    df_merge['SYMPTOMS'] = df_merge['TEXT'].apply(process_symptoms)
    df_merge


    # In[ ]:

    df_merge.to_csv("ProcessedData1.csv")

process(csvFileNoteEvents)

# In[ ]:




