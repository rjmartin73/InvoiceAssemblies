import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

df = pd.read_csv('./assets/assembly_conduit_wire_train.csv')

df = df[['Description', 'Label']]

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Description'].tolist(), df['Label'].tolist(), test_size=0.2)

# convert class labels to numbers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)

#Save label mapping for later
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(label_mapping)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize descriptions
train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                            max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, 
                          max_length=128)


class ConduitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, 
                val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    

# Convert encodings into datasets
train_dataset = ConduitDataset(train_encodings, train_labels)
val_dataset = ConduitDataset(val_encodings, val_labels)

num_labels = len(label_mapping) # Number of unique categories

model = BertForSequenceClassification.from_pretrained('bert-base-uncased'
                                                      , num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./bert_conduit_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()