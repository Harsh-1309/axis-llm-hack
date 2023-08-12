import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from sentence_transformers.readers import InputExample
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import gc
from accelerate import Accelerator
from accelerate import notebook_launcher
import os
import random
import hnswlib
from sklearn.model_selection import train_test_split

writer = SummaryWriter("runs/sgpt-bi")

# df_train=pd.read_csv("nfcorpus/final_data_train.csv")
# df_test=pd.read_csv("nfcorpus/final_data_test.csv")

df=pd.read_csv("jd_pos_neg.csv")

# print(df)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)


class CustomTextDataset(Dataset):
    def __init__(self, label, query, doc):
        self.labels = label
        self.query = query
        self.doc = doc
    def __len__(self):
            return len(self.query)
    def __getitem__(self, idx):
            label = self.labels[idx]
            query = self.query[idx]
            doc = self.doc[idx]
            sample = {"Query": query, "Doc": doc, "Label": label}
            return sample


# In[60]:


TD_train = CustomTextDataset(train_df['label'],train_df['title'],train_df['desc'])
DL_DS_train = DataLoader(TD_train, batch_size=4,shuffle=True)


# In[66]:


TD_val = CustomTextDataset(val_df['label'],val_df['title'],val_df['desc'])
DL_DS_val = DataLoader(TD_val, batch_size=4,shuffle=True)


# In[62]:


# next(iter(DL_DS_train))


# In[7]:


# all_docs=list(docs_df_test.Doc)
# all_docs_ids=list(docs_df_test.Did)
# all_queries=list(queries_df_test.Query)
# all_queries_ids=list(queries_df_test.Qid)


# In[8]:


def get_all_metrics(k,query_emb,doc_emb,all_queries_ids,all_docs_ids,rel_df):

    mrr_scores={}
    ndcg_scores={}
    pr={}
    re={}
    f1={}
    mrr_k=[10,20]
    ndcg_k=[10,30,50,100]
    prf_k=[10,30,50,100]

    ids=np.arange(len(doc_emb))
    p = hnswlib.Index(space = 'cosine', dim = len(doc_emb[0]))
    p.init_index(max_elements = len(doc_emb), ef_construction = 300, M = 32)
    p.add_items(doc_emb, ids)
    p.set_ef(350)
    
    def ndcg(true_relevance, pred_relevance, k_):
        if k_ is not None:
            true_relevance = true_relevance[:k_]
            pred_relevance = pred_relevance[:k_]

        dcg = np.sum((2 ** np.array(true_relevance) - 1) / np.log2(np.arange(2, len(true_relevance) + 2)))
        idcg = np.sum((2 ** np.sort(np.array(true_relevance))[::-1] - 1) / np.log2(np.arange(2, len(true_relevance) + 2)))

        return dcg / idcg if idcg > 0 else 0.0

    for i in range(len(all_queries_ids)):
        labels, distances = p.knn_query(query_emb[i], k = k)

        ## get doc ids
        ann_ids=[]
        for j in labels[0]:
            ann_ids.append(all_docs_ids[j])

        positive_samples=list(rel_df[rel_df.Query==all_queries_ids[i]].Doc)

        ## MRR
        for n in mrr_k:
            positive_sample_ranks=[]
            for an in ann_ids[:n]:
                if an in positive_samples:
                    positive_sample_ranks.append(ann_ids[:n].index(an)+1)
                    break
                else:
                    positive_sample_ranks.append(0)
            if n not in mrr_scores.keys():
                if positive_sample_ranks[0]!=0:
                    mrr_scores[n] = 1/positive_sample_ranks[0]
            else:
                if positive_sample_ranks[0]!=0:
                    mrr_scores[n] += 1/positive_sample_ranks[0]

        pred_score=[1-x for x in distances[0]]
        true_scores=[]
        

        ## nDCG
        for ann in ann_ids:
            if ann in positive_samples:
                true_scores.append(1)
            else:
                true_scores.append(0)

        for n in ndcg_k:
            # temp_scores=[]
            # temp_scores.append(ndcg(true_scores,pred_score,n))
            if n not in ndcg_scores.keys():
                ndcg_scores[n]=ndcg(true_scores,pred_score,n)
            else:
                ndcg_scores[n]+=ndcg(true_scores,pred_score,n)

        ## PRF
        for n in prf_k:
            count=0 
            # temp_p=[]
            temp_r=[]
            for ann in ann_ids[:n]:
                if ann in positive_samples:
                    count+=1

            if n not in pr.keys():
                pr[n]=count/n
            else:
                pr[n]+=count/n

            if n not in re.keys():
                re[n]=count/len(positive_samples)
            else:
                re[n]+=count/len(positive_samples)
    
    for n in mrr_k:
        try:
            mrr_scores[n]/=len(all_queries_ids)
        except KeyError:
            continue
    for n in ndcg_k:
        ndcg_scores[n]/=len(all_queries_ids)
    for n in prf_k:
        pr[n]/=len(all_queries_ids)
        re[n]/=len(all_queries_ids)
        f1[n]=2/((1/pr[n])+(1/re[n]))
    
    return mrr_scores, ndcg_scores, pr, re, f1


# In[9]:


tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")
model = AutoModel.from_pretrained("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")


# In[10]:


optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]


# In[11]:


def tokenize_with_specb(texts, is_query):
    # Tokenize without padding
    batch_tokens = tokenizer(texts, padding=False, truncation=True, max_length=300)   
    # Add special brackets & pay attention to them
    for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
        if is_query:
            seq.insert(0, SPECB_QUE_BOS)
            seq.append(SPECB_QUE_EOS)
        else:
            seq.insert(0, SPECB_DOC_BOS)
            seq.append(SPECB_DOC_EOS)
        att.insert(0, 1)
        att.append(1)
    # Add padding
    batch_tokens = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
    return batch_tokens


# In[12]:


def get_weightedmean_embedding(batch_tokens, model, state="train"):
    # Get the embeddings
    # with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
    # model=model.to(device).half()
    if state=="train":
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state
    else:
        batch_tokens.to('cuda')
        with torch.no_grad():
            last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

    # print(last_hidden_state)
    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings


# In[13]:


num_epochs=5
model_save_path="models/sgpt-bi"


# In[14]:


def calc_exp(q,d,t=10):
    q = q.unsqueeze(0)
    d = d.unsqueeze(0)
    return torch.exp(t * cosine_similarity(q, d))


# In[ ]:


def train(model=model,optimizer=optimizer,data_train=DL_DS_train):
    # loss=torch.nn.CosineEmbeddingLoss()
    # loss=losses.SupConLoss(temperature=10)
    # loss=ContrastiveLoss(margin=1)
    accelerator = Accelerator(mixed_precision="fp16")
    model, optimizer, DL_DS_train = accelerator.prepare(model, optimizer, data_train)

    
    # model.to(device)
    model.train()
    print_every=5
    evaluate_after=500
    query_infer_batch=50
    doc_infer_batch=50
    model_loc = "models/sgpt-bi"
    if not os.path.isdir(model_loc):
        os.makedirs(model_loc)

    step = 0
    running_loss=0.0
    for e in tqdm(range(1,num_epochs+1)):
        print(f"training epoch {e}")
        model.train()
        for _, data_ in enumerate(tqdm(DL_DS_train)):
            step += 1
            optimizer.zero_grad()       

            query_=[x for x in data_["Query"]]
            doc_=[x for x in data_["Doc"]]
            label_=[x for x in data_["Label"]]
            label_=torch.tensor(label_).to(accelerator.device)
            
            query_tokens=tokenize_with_specb(query_, is_query=True).to(accelerator.device)
            doc_tokens=tokenize_with_specb(doc_, is_query=False).to(accelerator.device) #td

            query_embed=get_weightedmean_embedding(query_tokens,model,"train")
            doc_embed=get_weightedmean_embedding(doc_tokens,model,"train")

            query_embed=nn.functional.normalize(query_embed, dim=1)
            doc_embed=nn.functional.normalize(doc_embed, dim=1)
            # print(query_embed[0])
            
            similarity_scores = torch.mm(query_embed, doc_embed.t())
            # batch_loss=0.0
            # Calculate cross-entropy loss
            loss = F.cross_entropy(similarity_scores, label_)
            
            ## extract hard 

            batch_loss=loss.mean()

            accelerator.backward(batch_loss)
            optimizer.step()

            # _, predicted = torch.max(similarity_scores.data, 1)
            # correct_predictions += (predicted == labels).sum().item()

            running_loss += batch_loss.item()
            if (step % print_every) == 0:
                writer.add_scalar("training loss",running_loss/print_every,step)
                running_loss=0.0

            if (step % evaluate_after) ==0:
                print(f"evaluating at step {step}")

                model.eval()
                total_val_loss = 0.0
                correct_val_predictions = 0
                with torch.no_grad():
                    # for batch in DL_DS_val:
                    step_val=0
                    running_loss_val=0.0
                    for _, data_val in enumerate(tqdm(DL_DS_val)):
                        step_val+=1
                        query_val=[x for x in data_val["Query"]]
                        doc_val=[x for x in data_val["Doc"]]
                        label_val=[x for x in data_val["Label"]]
                        label_val=torch.tensor(label_val).to(accelerator.device)

                        query_tokens_val=tokenize_with_specb(query_val, is_query=True).to(accelerator.device)
                        doc_tokens_val=tokenize_with_specb(doc_val, is_query=False).to(accelerator.device) #td

                        query_embed_val=get_weightedmean_embedding(query_tokens_val,model,"test")
                        doc_embed_val=get_weightedmean_embedding(doc_tokens_val,model,"test")

                        query_embed_val=nn.functional.normalize(query_embed_val, dim=1)
                        doc_embed_val=nn.functional.normalize(doc_embed_val, dim=1)

                        similarity_scores_val = torch.mm(query_embed_val, doc_embed_val.t())
                        # batch_loss=0.0
                        # Calculate cross-entropy loss
                        loss_val = F.cross_entropy(similarity_scores_val, label_val)
                        loss_val=loss_val.mean()

                        _, val_predicted = torch.max(similarity_scores_val.data, 1)
                        correct_val_predictions += (val_predicted == label_val).sum().item()

                        running_loss_val += loss_val.item()
                        if (step_val % print_every) == 0:
                            writer.add_scalar("validation loss",running_loss/print_every,step_val)
                            loss_val=0.0
                # avg_val_loss = loss_val.mean()
                # val_accuracy = sum(correct_val_predictions)/len(correct_val_predictions)
                # writer.add_scalar("Validation accuracy",val_accuracy,step_val)
 
        print(f"Model saved at - {model_save_path}/model_{e}.pt")
        torch.save(model.state_dict(), f"{model_save_path}/model_{e}.pt")

gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()

# notebook_launcher(train,num_processes=1)
if __name__ == '__main__':
    train()