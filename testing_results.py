from transformers import pipeline

# Load trained model
classifier = pipeline("text-classification", model="./bert_conduit_model", tokenizer="./bert_conduit_model")

# Label mapping (Adjust based on how your dataset was labeled)
label_mapping = {0: "Conduit", 2: "Wire", 1: "Exclude"}

# Test with a description
description = "PVC BELL END 1-1/4"
prediction = classifier(description)[0]  # Extract first result

# Convert label to category name
category = label_mapping[int(prediction["label"].split("_")[-1])]  # Convert 'LABEL_0' to integer and map

# Output result
print(f"Description: {description}")
print(f"Predicted Category: {category} (Confidence: {prediction['score']:.4f})")

descriptions = ["PVC BELL END 1-1/4", "SCH80 BELL END 3 INCH", "EMT 3/4 COUPLING"]
predictions = classifier(descriptions)

for desc, pred in zip(descriptions, predictions):
    print(f"{desc} → {pred['label']} (Confidence: {pred['score']:.4f})")
