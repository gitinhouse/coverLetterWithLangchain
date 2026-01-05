from transformers import AutoModelForSequenceClassification, AutoTokenizer

def downloadModel():
    model_name = "Falconsai/intent_classification"
    save_directory = "./ModelForValidation"

    # This downloads and saves the model/tokenizer locally
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    print(f"Model saved to {save_directory}")

if __name__ == "__main__":
    downloadModel()