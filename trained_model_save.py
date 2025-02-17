import pickle
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Function to save label encoder
def save_label_encoder(label_encoder, filename):
    with open(filename, "wb") as f:
        pickle.dump(label_encoder, f)

# Function to save the model
def save_model(model, model_path):
    model.save_pretrained(model_path)

# Example: Conduit Type Model
conduit_type_labels = ["CONDUIT - EMT", "CONDUIT - PVC", "CONDUIT - GRC", "CONDUIT - FLEX", "CONDUIT - ENT", "UNK"]
conduit_type_encoder = LabelEncoder()
conduit_type_encoded = conduit_type_encoder.fit_transform(conduit_type_labels)

# Train Conduit Type Model (dummy example)
model_conduit_type = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(conduit_type_labels))

# Save encoder and model
save_label_encoder(conduit_type_encoder, "conduit_type_label_encoder.pkl")
save_model(model_conduit_type, "./models/conduit_type_model")

# Example: Conduit Size Model
conduit_size_labels = ['1/2"', '3/4"', '1"', '1-1/4"', '1-1/2"', '1-3/4"', '2"', '2-1/2"', '3"', '3-1/2"', '4"']
conduit_size_encoder = LabelEncoder()
conduit_size_encoded = conduit_size_encoder.fit_transform(conduit_size_labels)

# Train Conduit Size Model (dummy example)
model_conduit_size = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(conduit_size_labels))



# Save encoder and model
save_label_encoder(conduit_size_encoder, "conduit_size_label_encoder.pkl")
save_model(model_conduit_size, "./models/conduit_size_model")

# Wire gauge model
wire_gauge_labels = ['#4/0', '#3/0', '#2/0', '#1/0', '#750', '#600', '#500', '#400', '#350', '#300', '#250', '#14',
                    '#12', '#10', '#8', '#6', '#4', '#3', '#2', '#1,', 'LOW VOLTAGE']
wire_gauge_encoder = LabelEncoder()
wire_gauge_encoded = wire_gauge_encoder.fit_transform(wire_gauge_labels)

model_wire_gauge = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(wire_gauge_labels))

save_label_encoder(wire_gauge_encoder, "wire_gauge_encoder.pkl")
save_model(model_wire_gauge, "./models/wire_gauge_model")

print("✅ All models and encoders saved successfully!")
