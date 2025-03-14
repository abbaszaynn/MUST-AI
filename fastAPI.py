from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI()

# Load the trained model and tokenizer
MODEL_PATH = os.getenv('MODEL_PATH', './model.SAFETENSORS')  # Path to your model directory

try:
    # Load model and tokenizer from the folder
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        inputs = tokenizer(input_data.text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_label].item()

        return {"label": predicted_label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Model API is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("render_fastapi_app:app", host="0.0.0.0", port=8000)
