from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load the trained model and tokenizer
MODEL_PATH = "./model.safetensors"
CONFIG_PATH = "./config.json"
TOKENIZER_PATH = "./vocab.txt"


try:
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG_PATH)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
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
    uvicorn.run("fastAPI:app", host="0.0.0.0", port=8000)

