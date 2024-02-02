from flask import Flask, render_template, request
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)

checkpoint = "Salesforce/codet5p-220m-bimodal"
device = "cpu"  

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    code = request.form['user_code']
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    generated_ids = model.generate(input_ids, max_length=20)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output

if __name__ == '__main__':
    app.run(debug=True)
