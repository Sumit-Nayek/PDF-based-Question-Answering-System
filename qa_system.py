# Updated qa_system.py for Inference API
from huggingface_hub import InferenceClient
import os
from config import HUGGINGFACEHUB_API_TOKEN

class QASystem:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.client = InferenceClient(token=HUGGINGFACEHUB_API_TOKEN)
        self.model_name = model_name
    
    def answer_question(self, context, question):
        response = self.client.question_answering(
            context=context,
            question=question,
            model=self.model_name
        )
        return response['answer']
