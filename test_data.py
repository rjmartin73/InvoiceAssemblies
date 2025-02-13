import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import pipeline
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load trained model
    classifier = pipeline("text-classification", model="./bert_conduit_model")

    # Load test data
    df_test = pd.read_csv("./assets/test_data.csv")

    # Define label mapping (adjust based on training)
    label_mapping = { 
        "LABEL_0": "Conduit",
        "LABEL_1": "Wire",
        "LABEL_2": "Exclude"
    }

    # Run predictions on all descriptions
    df_test["PredictedLabel"] = df_test["Description"].apply(lambda x: label_mapping[classifier(x)[0]['label']])

    # Ensure all expected labels are present
    labels = ["Conduit", "Wire", "Exclude"]

    # Generate confusion matrix
    cm = confusion_matrix(df_test["ActualLabel"], df_test["PredictedLabel"], labels=labels)

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Print classification report (precision, recall, F1-score)
    print(classification_report(df_test["ActualLabel"], df_test["PredictedLabel"], labels=labels, zero_division=1))

    return (
        classification_report,
        classifier,
        cm,
        confusion_matrix,
        df_test,
        label_mapping,
        labels,
        pd,
        pipeline,
        plt,
        sns,
    )


@app.cell
def _(pd):

    df_train = pd.read_csv("./assets/random_sample_classified_items.csv")
    print(df_train["Label"].value_counts())

    return (df_train,)


if __name__ == "__main__":
    app.run()
