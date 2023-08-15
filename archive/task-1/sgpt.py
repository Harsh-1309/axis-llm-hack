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

train_df=pd.read_csv("jd_pos_neg.csv")

# print(df)
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# train_df.reset_index(drop=True, inplace=True)
# val_df.reset_index(drop=True, inplace=True)


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


TD_train = CustomTextDataset(train_df['label'],train_df['title'],train_df['desc'])
DL_DS_train = DataLoader(TD_train, batch_size=4,shuffle=True)


tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")
model = AutoModel.from_pretrained("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]


def tokenize_with_specb(texts, is_query):
    # Tokenize without padding
    batch_tokens = tokenizer(texts, padding=False, truncation=True, max_length=256)   
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

num_epochs=3
model_save_path="models/sgpt-bi"



def train(model=model,optimizer=optimizer,data_train=DL_DS_train):
    # loss=torch.nn.CosineEmbeddingLoss()
    # loss=losses.SupConLoss(temperature=10)
    # loss=ContrastiveLoss(margin=1)
    # accelerator = Accelerator(mixed_precision="fp16")
    accelerator = Accelerator()
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

            # if (step % evaluate_after) ==0:
            #     print(f"evaluating at step {step}")

            #     model.eval()
            #     total_val_loss = 0.0
            #     correct_val_predictions = 0
            #     with torch.no_grad():
            #         # for batch in DL_DS_val:
            #         step_val=0
            #         running_loss_val=0.0
            #         for _, data_val in enumerate(tqdm(DL_DS_val)):
            #             step_val+=1
            #             query_val=[x for x in data_val["Query"]]
            #             doc_val=[x for x in data_val["Doc"]]
            #             label_val=[x for x in data_val["Label"]]
            #             label_val=torch.tensor(label_val).to(accelerator.device)

            #             query_tokens_val=tokenize_with_specb(query_val, is_query=True).to(accelerator.device)
            #             doc_tokens_val=tokenize_with_specb(doc_val, is_query=False).to(accelerator.device) #td

            #             query_embed_val=get_weightedmean_embedding(query_tokens_val,model,"test")
            #             doc_embed_val=get_weightedmean_embedding(doc_tokens_val,model,"test")

            #             query_embed_val=nn.functional.normalize(query_embed_val, dim=1)
            #             doc_embed_val=nn.functional.normalize(doc_embed_val, dim=1)

            #             similarity_scores_val = torch.mm(query_embed_val, doc_embed_val.t())
            #             # batch_loss=0.0
            #             # Calculate cross-entropy loss
            #             loss_val = F.cross_entropy(similarity_scores_val, label_val)
            #             loss_val=loss_val.mean()

            #             _, val_predicted = torch.max(similarity_scores_val.data, 1)
            #             correct_val_predictions += (val_predicted == label_val).sum().item()

            #             running_loss_val += loss_val.item()
            #             if (step_val % print_every) == 0:
            #                 writer.add_scalar("validation loss",running_loss/print_every,step_val)
            #                 loss_val=0.0
            #     # avg_val_loss = loss_val.mean()
            #     # val_accuracy = sum(correct_val_predictions)/len(correct_val_predictions)
            #     # writer.add_scalar("Validation accuracy",val_accuracy,step_val)
 
        print(f"Model saved at - {model_save_path}/model_{e}.pt")
        torch.save(model.state_dict(), f"{model_save_path}/model_{e}.pt")

gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()

# notebook_launcher(train,num_processes=1)
if __name__ == '__main__':
    train()