from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pickle

class IntentClassifierService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:   
            cls._instance = super(IntentClassifierService, cls).__new__(cls)
            cls._instance.load_intent_classifier_model()
            print("Initialized Singleton Intent")
        return cls._instance
    
    def __call__(self):
        return self

    def load_intent_classifier_model(self):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("./intent-classifier-model/") #Define BERT model on initial run
            self.tokenizer = AutoTokenizer.from_pretrained("./query-tokenizer/") #Define bert tokenizer on initialrun
            self.le = pickle.load(open("./label_encoder.pkl", "rb"))
        except Exception as e:
            print(f"Error loading intent classifier model: {e}")
            self.model = None
            self.tokenizer = None

    def predict_intent(self, text: str):
        encoded = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**encoded).logits
            
        predicted_class_id = logits.argmax().item()
        return self.le.classes_[predicted_class_id]


def get_intent_classifier_service():
    intent_classifier_service = IntentClassifierService()
    try:
        yield intent_classifier_service
    except Exception as e:
        print(f"Error getting intent classifier service: {e}")
        return None
    