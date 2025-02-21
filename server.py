from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the Hugging Face Model
MODEL_NAME = "Kargichauhan/Llama-3-MentalHealth"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get("input_text", "")

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=100)

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

