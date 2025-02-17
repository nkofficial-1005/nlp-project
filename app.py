import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_path = "./models/ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def ner_prediction(text):
    entities = ner_pipeline(text)
    return {e["word"]: e["entity"] for e in entities}

# Gradio UI
iface = gr.Interface(fn=ner_prediction, inputs="text", outputs="label")
iface.launch(server_name="0.0.0.0", server_port=7860)