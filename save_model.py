from transformers import BertForSequenceClassification, BertTokenizer

# Specify the best checkpoint (Update the path if needed)
best_checkpoint = "./bert_conduit_model/checkpoint-845"  # Update based on your highest checkpoint

# Load model from best checkpoint
model = BertForSequenceClassification.from_pretrained(best_checkpoint)

# Fix tokenizer by loading from the original BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # ✅ Fixing tokenizer issue

# Save the model and tokenizer properly
model.save_pretrained("./bert_conduit_model")
tokenizer.save_pretrained("./bert_conduit_model")

print("Model and tokenizer saved successfully!")
