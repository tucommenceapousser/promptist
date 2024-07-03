from flask import Flask, request, render_template_string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

def load_prompter():
    prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return prompter_model, tokenizer

prompter_model, prompter_tokenizer = load_prompter()

def generate(plain_text):
    inputs = prompter_tokenizer(plain_text.strip() + " Rephrase:", return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    eos_id = prompter_tokenizer.eos_token_id

    outputs = prompter_model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=75,
        num_beams=8,
        num_return_sequences=8,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        length_penalty=-1.0
    )

    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text + " Rephrase:", "").strip()
    return res

@app.route('/')
def home():
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta name="description" content="Promptist est une interface de génération de texte optimisée pour Stable Diffusion v1-4 par TRHACKNON.">
            <meta name="keywords" content="GPT-4, génération de texte, Stable Diffusion, IA, Promptist, TRHACKNON">
            <meta name="author" content="TRHACKNON">
            <meta property="og:title" content="Promptist Demo by TRHACKNON">
            <meta property="og:description" content="Utilisez Promptist pour générer des textes optimisés pour Stable Diffusion v1-4. Découvrez la puissance de l'IA avec TRHACKNON.">
            <meta property="og:image" content="https://static-trkn.replit.app/cup.png">
            <meta property="og:url" content="https://promptist-trkn.replit.app/">
            <meta name="twitter:card" content="summary_large_image">
            <meta property="twitter:title" content="Promptist Demo by TRHACKNON">
            <meta property="twitter:description" content="Utilisez Promptist pour générer des textes optimisés pour Stable Diffusion v1-4. Découvrez la puissance de l'IA avec TRHACKNON.">
            <meta property="twitter:image" content="https://static-trkn.replit.app/cup.png">
            <meta property="twitter:url" content="https://promptist-trkn.replit.app/">
            <link rel="stylesheet" href="https://static-trkn.replit.app/styles.css">
            <title>Promptist Demo by TRHACKNON</title>
            <style>
                body {
                    background-color: black;
                    color: #00FF00;
                    font-family: 'Courier New', Courier, monospace;
                }
                h1 {
                    color: #FF4500;
                }
                .container {
                    width: 80%;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #1E1E1E;
                    border-radius: 10px;
                    box-shadow: 0 0 10px #00FF00;
                }
                textarea, input[type="submit"] {
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #00FF00;
                    border-radius: 5px;
                    background-color: #333333;
                    color: #00FF00;
                    font-size: 1em;
                }
                input[type="submit"] {
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #FF4500;
                    color: black;
                }
                .logo {
                  width: 100px;
                  height: auto;
                }
                .logo-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
            <div class="logo-container">
            <img src="https://static-trkn.replit.app/trkn.svg" alt="trhacknon Logo" class="logo">
            </div>
                <h1>Promptist Demo by TRHACKNON</h1>
                <p>Promptist is a prompt interface for Stable Diffusion v1-4 that optimizes user input into model-preferred prompts. The online demo at Hugging Face Spaces is using CPU, so slow generation speed would be expected. Please load the model locally with GPUs for faster generation.</p>
                <form action="/generate" method="post">
                    <textarea name="input_text" rows="4" placeholder="Input Prompt"></textarea><br>
                    <input type="submit" value="Generate">
                </form>
                {% if output_text %}
                    <h2>Optimized Prompt</h2>
                    <p>{{ output_text }}</p>
                {% endif %}
            </div>
        </body>
        </html>
    ''', output_text=None)

@app.route('/generate', methods=['POST'])
def generate_text():
    input_text = request.form['input_text']
    output_text = generate(input_text)
    return render_template_string('''
        <!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <html lang="en">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Promptist est une interface de génération de texte optimisée pour Stable Diffusion v1-4 par TRHACKNON.">
    <meta name="keywords" content="GPT-4, génération de texte, Stable Diffusion, IA, Promptist, TRHACKNON">
    <meta name="author" content="TRHACKNON">
    <meta property="og:title" content="Promptist Demo by TRHACKNON">
    <meta property="og:description" content="Utilisez Promptist pour générer des textes optimisés pour Stable Diffusion v1-4. Découvrez la puissance de l'IA avec TRHACKNON.">
    <meta property="og:image" content="https://static-trkn.replit.app/cup.png">
    <meta property="og:url" content="https://promptist-trkn.replit.app/">
    <meta name="twitter:card" content="summary_large_image">
    <meta property="twitter:title" content="Promptist Demo by TRHACKNON">
    <meta property="twitter:description" content="Utilisez Promptist pour générer des textes optimisés pour Stable Diffusion v1-4. Découvrez la puissance de l'IA avec TRHACKNON.">
    <meta property="twitter:image" content="https://static-trkn.replit.app/cup.png">
    <meta property="twitter:url" content="https://promptist-trkn.replit.app/">
    <link rel="stylesheet" href="https://static-trkn.replit.app/styles.css">
            <title>Promptist Demo by TRHACKNON</title>
            <style>
                body {
                    background-color: black;
                    color: #00FF00;
                    font-family: 'Courier New', Courier, monospace;
                }
                h1 {
                    color: #FF4500;
                }
                .container {
                    width: 80%;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #1E1E1E;
                    border-radius: 10px;
                    box-shadow: 0 0 10px #00FF00;
                }
                textarea, input[type="submit"] {
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #00FF00;
                    border-radius: 5px;
                    background-color: #333333;
                    color: #00FF00;
                    font-size: 1em;
                }
                input[type="submit"] {
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #FF4500;
                    color: black;
                }
                .logo {
                  width: 100px;
                  height: auto;
                }
                .logo-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
            <div class="logo-container">
            <img src="https://static-trkn.replit.app/trkn.svg" alt="trhacknon Logo" class="logo">
            </div>
                <h1>Promptist Demo by TRHACKNON</h1>
                <p>Promptist is a prompt interface for Stable Diffusion v1-4 that optimizes user input into model-preferred prompts. The online demo at Hugging Face Spaces is using CPU, so slow generation speed would be expected. Please load the model locally with GPUs for faster generation.</p>
                <form action="/generate" method="post">
                    <textarea name="input_text" rows="4" placeholder="Input Prompt">{{ input_text }}</textarea><br>
                    <input type="submit" value="Generate">
                </form>
                <h2>Optimized Prompt</h2>
                <p>{{ output_text }}</p>
            </div>
        </body>
        </html>
    ''', input_text=input_text, output_text=output_text)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)