import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json
import logging
import random

# Define the labels
LABELS = ["Extraction", "Filling", "Root Canal", "Cleaning", "Initial Checkup"]
NUM_LABELS = len(LABELS)

logger = logging.getLogger(__name__)

class DentalClassifier:
    def __init__(self, model_path=None):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        if model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-multilingual-cased',
                num_labels=NUM_LABELS
            )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess_data(self, texts, labels=None):
        # Lowercase all texts for consistency
        texts = [t.lower() for t in texts]
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        if labels is not None:
            return Dataset.from_dict({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels
            })
        return encodings

    def train(self, train_texts, train_labels, eval_texts=None, eval_labels=None, output_dir='./results'):
        # Shuffle the data for better generalization
        combined = list(zip(train_texts, train_labels))
        random.shuffle(combined)
        train_texts[:], train_labels[:] = zip(*combined)
        # Prepare datasets
        train_dataset = self.preprocess_data(train_texts, train_labels)
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_dataset = self.preprocess_data(eval_texts, eval_labels)

        # Define training arguments with more epochs
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            no_cuda=not torch.cuda.is_available(),
            report_to="none"
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()

        # Save the model and tokenizer
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save model configuration
            config = {
                'num_labels': NUM_LABELS,
                'labels': LABELS
            }
            with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Model saved successfully to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            # Lowercase input for consistency
            inputs = self.preprocess_data([text.lower()])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            return LABELS[predicted_class]

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

# Example usage:
if __name__ == "__main__":
    # New, clean, non-overlapping, and distinct dataset for each class
    extraction_examples = [
        "My tooth is loose and needs to be pulled.",
        "I need to have my tooth removed.",
        "My wisdom tooth is causing problems and should be extracted.",
        "My tooth is broken and cannot be saved.",
        "I have a tooth that must be taken out.",
        "My dentist said I need a tooth extraction.",
        "My tooth is fractured below the gum line.",
        "I have a tooth that is beyond repair and needs removal.",
        "My tooth is severely decayed and must be pulled.",
        "I have a tooth that is making my jaw swell and needs to be removed.",
        "My tooth is causing my cheek to swell and needs extraction.",
        "My tooth is infected and the dentist recommended extraction.",
        "I have a tooth that is hurting and wobbly, needs to be pulled.",
        "My tooth is broken to the gum line and can't be fixed.",
        "I have a tooth that is loose after an accident.",
        "My tooth is causing pain when I bite and needs to be removed.",
        "My tooth is black and painful, dentist said to extract it.",
        "I have a tooth that is making it hard to eat and needs extraction.",
        "My tooth is causing headaches and must be pulled.",
        "I have a tooth that is decayed to the root and needs to be removed.",
        "My tooth is causing gum swelling and needs extraction.",
        "I have a tooth that is loose and bleeding, dentist said to pull it.",
        "My tooth is broken and sharp, dentist recommended extraction.",
        "I have a tooth that is sensitive to pressure and must be removed.",
        "My tooth is causing my face to swell and needs to be pulled.",
        "I have a tooth that is infected and smells, dentist said to extract.",
        "My tooth is hurting all the time and needs to be removed.",
        "My tooth is cracked in half and can't be saved.",
        "I have a tooth that is making my gums bleed and needs extraction.",
        "My tooth is causing pain in my mouth and must be pulled.",
        "I have a tooth that is making it hard to sleep and needs to be removed.",
        "My tooth is broken and needs extraction.",
        "I have a tooth that is causing pain in my head and must be pulled.",
        "My tooth is loose and making it hard to eat, dentist said to extract.",
        "My tooth is infected and needs removal.",
        "My tooth is broken and can't be repaired, needs extraction.",
        "My tooth is causing pain when I drink water and must be pulled.",
        "My tooth is loose and making my gums bleed, dentist said to extract.",
        "My tooth is broken and causing swelling, dentist recommended extraction.",
        "My tooth is hurting and needs to be pulled.",
        "My tooth is decayed and can't be saved, dentist said to extract.",
        "My tooth is causing pain when I chew food and must be removed.",
        "My tooth is loose and making my mouth hurt, dentist said to extract.",
        "My tooth is broken and making it hard to talk, needs extraction.",
        "My tooth is causing pain in my neck and must be pulled.",
        "My tooth is hurting and making my gums swell, dentist said to extract.",
        "My tooth is cracked and making my jaw hurt, dentist recommended extraction.",
        "My tooth is broken and making my cheek swell, dentist said to extract.",
        "My tooth is causing pain when I smile and must be pulled."
    ]
    filling_examples = [
        "I have a cavity in my tooth.",
        "My tooth hurts when I eat sweets.",
        "I need a filling for my tooth.",
        "There is a small hole in my tooth.",
        "My tooth is sensitive to cold drinks.",
        "I have a decayed tooth that needs a filling.",
        "My tooth is chipped and needs a filling.",
        "My tooth hurts when I eat chocolate.",
        "I have a tooth with a cavity that needs to be filled.",
        "My tooth is hurting and has a small hole.",
        "I need a dental filling for my molar.",
        "My tooth is sensitive to hot and cold.",
        "I have a tooth that is decayed and needs a filling.",
        "My tooth hurts when I drink cold water.",
        "I have a tooth that is chipped and painful.",
        "My tooth is hurting and needs a filling.",
        "I have a cavity in my front tooth.",
        "My tooth is sensitive to sweets.",
        "I have a tooth that is hurting when I eat.",
        "My tooth is chipped and needs repair.",
        "My tooth is decayed and painful, needs a filling.",
        "I have a tooth that is hurting when I drink juice.",
        "My tooth is sensitive to ice cream.",
        "I have a tooth that is chipped and needs a dental filling.",
        "My tooth is hurting and needs to be filled.",
        "I have a tooth that is decayed and needs repair.",
        "My tooth hurts when I eat candy.",
        "I have a tooth that is chipped and sensitive.",
        "My tooth is hurting and has a cavity that needs filling.",
        "I have a tooth that is decayed and needs to be filled.",
        "My tooth hurts when I eat something cold.",
        "I have a tooth that is chipped and needs a filling.",
        "My tooth is hurting and has a cavity that needs repair.",
        "I have a tooth that is decayed and needs a filling.",
        "My tooth hurts when I eat something hot.",
        "I have a tooth that is chipped and needs a dental filling.",
        "My tooth is hurting and has a cavity that needs to be filled.",
        "I have a tooth that is decayed and needs repair.",
        "My tooth hurts when I eat something sweet.",
        "I have a tooth that is chipped and needs to be filled.",
        "My tooth is hurting and has a cavity that needs filling."
    ]
    root_canal_examples = [
        "My tooth pain is deep and throbbing, especially at night.",
        "I have pain radiating from my tooth to my ear.",
        "My tooth is sensitive to hot and cold and the pain is severe.",
        "I have a severe toothache that keeps me awake.",
        "My tooth hurts badly and the pain is deep.",
        "My tooth is throbbing and keeps me awake.",
        "I have a tooth that hurts when I bite down and the pain lingers.",
        "My tooth is aching and the pain is constant.",
        "I have a tooth that is sensitive to pressure and the pain is sharp.",
        "My tooth is hurting and the pain is intense and deep.",
        "I have a tooth that is causing pain in my jaw and ear.",
        "My tooth is hurting and the pain is severe and constant.",
        "I have a tooth that is sensitive to touch and pain is deep.",
        "My tooth is hurting and the pain is intense and severe.",
        "I have a tooth that is causing pain in my head and jaw.",
        "My tooth is hurting and the pain is constant and severe.",
        "I have a tooth that is sensitive to pressure and touch and pain is deep.",
        "My tooth is hurting and the pain is sharp and intense.",
        "I have a tooth that is causing pain in my jaw and head.",
        "My tooth is hurting and the pain is severe and deep.",
        "I have a tooth that is sensitive to touch and pressure and pain is deep.",
        "My tooth is hurting and the pain is intense and constant.",
        "I have a tooth that is causing pain in my ear and jaw.",
        "My tooth is hurting and the pain is deep and severe.",
        "My tooth is sensitive to hot and cold and touch and pain is deep.",
        "My tooth is hurting and the pain is throbbing and intense.",
        "I have a tooth that is causing pain in my head and ear.",
        "My tooth is hurting and the pain is constant and deep.",
        "I have a tooth that is sensitive to pressure and hot and pain is deep.",
        "My tooth is hurting and the pain is sharp and severe.",
        "I have a tooth that is causing pain in my jaw and ear and head.",
        "My tooth is hurting and the pain is severe and intense.",
        "I have a tooth that is sensitive to touch and hot and pain is deep.",
        "My tooth is hurting and the pain is intense and severe.",
        "I have a tooth that is causing pain in my ear and jaw and head.",
        "My tooth is hurting and the pain is deep and intense."
    ]
    cleaning_examples = [
        "I want a dental cleaning.",
        "I need my teeth cleaned.",
        "I want to remove tartar from my teeth.",
        "I want a professional cleaning.",
        "I want to whiten my teeth.",
        "cleaning",
        "I want to polish my teeth.",
        "I want to get rid of stains on my teeth.",
        "I want a scaling and polishing.",
        "I want to book a cleaning appointment.",
        "I want to get my teeth cleaned professionally.",
        "I want to remove plaque from my teeth.",
        "I want to have my teeth cleaned.",
        "I want to get a dental cleaning appointment.",
        "I want to clean my teeth thoroughly.",
        "I want to have my teeth professionally cleaned.",
        "I want to get my teeth scaled and polished.",
        "I want to clean my teeth deeply.",
        "I want to get my teeth cleaned at the dentist.",
        "I want to remove yellow stains from my teeth.",
        "I want to get my teeth cleaned and polished.",
        "I want to have my teeth cleaned and whitened.",
        "I want to get my teeth cleaned and scaled.",
        "I want to have my teeth cleaned and polished.",
        "I want to get my teeth cleaned and whitened.",
        "I want to have my teeth cleaned and scaled.",
        "I want to get my teeth cleaned and polished at the dentist.",
        "I want to have my teeth cleaned and whitened at the dentist.",
        "I want to get my teeth cleaned and scaled at the dentist.",
        "I want to have my teeth cleaned and polished at the dentist."
    ]
    checkup_examples = [
        "I want a dental checkup.",
        "I need a routine dental exam.",
        "I want to see a dentist for a checkup.",
        "I want to make sure my teeth are healthy.",
        "I want a regular dental checkup.",
        "I want to schedule a dental exam.",
        "I want to get my teeth checked.",
        "I want to have a dental checkup.",
        "I want to book a dental checkup.",
        "I want to get a dental exam.",
        "I want to have my teeth examined.",
        "I want to see a dentist for a routine checkup.",
        "I want to get a dental checkup appointment.",
        "I want to have a dental exam appointment.",
        "I want to get my teeth checked by a dentist.",
        "I want to have my teeth examined by a dentist.",
        "I want to see a dentist for a dental checkup.",
        "I want to get a dental checkup and cleaning.",
        "I want to have a dental checkup and cleaning.",
        "I want to get a dental exam and cleaning."
    ]
    train_texts = extraction_examples + filling_examples + root_canal_examples + cleaning_examples + checkup_examples
    train_labels = [0]*len(extraction_examples) + [1]*len(filling_examples) + [2]*len(root_canal_examples) + [3]*len(cleaning_examples) + [4]*len(checkup_examples)

    # Initialize and train the model
    classifier = DentalClassifier()
    classifier.train(train_texts, train_labels)

    # Example prediction
    test_text = "I want a dental cleaning"
    prediction = classifier.predict(test_text)
    print(f"Predicted treatment: {prediction}")
