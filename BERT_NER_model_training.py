import torch
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric


# In[3]:


### check pytorch and transformers version
print("Check Transformers and Pytorch version")
print(transformers.__version__)
print(torch.__version__)


# #### Model setting

# In[4]:


### set task "ner" = name entity recognition
task = "ner"

### specify pre-trained model
model_checkpoint = "bert-base-chinese"

batch_size = 16
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


# #### Import training data

# In[5]:


data = np.load('./Training_data_cleaned/cleaned_data_np.npy',allow_pickle=True)
print("pre-tagged data imported")

token = data[0]
token_list = [sentence.tolist() for sentence in token]


# In[6]:


### read data from pre-tagged data
unique_tags = set(tag for doc in data[1] for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

### unique tags from pre-tagged data
label_list=list(unique_tags)


# ### Tokenizing
# #### training data tokenized

# In[7]:


### Setting training set size/total data set(# total 3000)
cut_off_num = 2700 


# In[8]:


train_tags = data[1][:cut_off_num] #set of label data (training)
val_tags = data[1][cut_off_num:] #set of label data (validation)


# In[9]:


###initialize tokenization (training data)
train_tokenized = tokenizer(token_list[:cut_off_num], is_split_into_words=True,return_offsets_mapping= True,\
                            padding=True, truncation=True)  

###initialize tokenization (validation data)
val_tokenized = tokenizer(token_list[cut_off_num:], is_split_into_words=True,return_offsets_mapping= True,\
                          padding=True, truncation=True)  


# In[10]:


###function to 1.combine raw text, 2. embedding and 3. offset mapping
def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    #print(labels)
    encoded_labels = []
    num = 0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        num+=1
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
        #print(doc_enc_labels.tolist())

    return encoded_labels


# In[11]:


### create tagged train and validate data set
train_labels = encode_tags(train_tags, train_tokenized)
val_labels = encode_tags(val_tags, val_tokenized)


# In[12]:


### NER object class
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[13]:


### remove offset mapping from data set
train_tokenized.pop("offset_mapping") # we don't want to pass this to the model
val_tokenized.pop("offset_mapping")

### create NER dataset object for both train and validation data set
train_dataset = NERDataset(train_tokenized, train_labels) #training object
val_dataset = NERDataset(val_tokenized, val_labels) #validation object

print("NER dataset successfully build")


# #### Modeling accuracy examination function

# In[14]:


### load validation metric "seqeval"
metric = load_metric("seqeval")

### validation function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# #### Import related BERT functions (BERT task -> Tokenclassification)

# In[ ]:


from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))


# ### Training

# In[ ]:


print("Start training")

training_args = TrainingArguments(
    evaluation_strategy= "steps",
    output_dir='./results',          # output directory
    num_train_epochs=30,             # total number of training epochs
    per_device_train_batch_size=20,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    load_best_model_at_end = True    #early stopping related

)

trainer = Trainer(
    model=model,                         # the instantiated   Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
    callbacks= [transformers.EarlyStoppingCallback(early_stopping_patience= 20)]    #early stopping benchmark
)

trainer.train()


# #### Trained model evaluation

# In[ ]:


### print evaluation
trainer.evaluate()


# #### Saving model

# In[ ]:


model_name = 'bert-base-chinese-carrefour_test'  #model name
model.save_pretrained(f"./model/{model_name}")
tokenizer.save_pretrained(f'./model/{model_name}')

print("Model successfully saved")

