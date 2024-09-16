import gradio as gr
from PIL import Image
import sys 
import pickle
import torch
import plotly.graph_objects as go
import torch.nn as nn
import string
from torchvision import transforms
from model import VQAClassifier 
from FeatureExtractor import extract_image_features_inst, extract_text_features
import matplotlib.pyplot as plt

sys.path.insert(1, "ClassificationModel")

input_size = 768
num_classes = 1548
dropout_prob = 0.2
num_heads = 4

model = VQAClassifier(embed_dim=input_size, num_classes=num_classes, dropout=dropout_prob, num_heads=num_heads)

state_dict = torch.load("Results/vqa_model_8h_0.2dropout_final.pth", map_location=torch.device('cpu')).get('model_state_dict')

if 'module' in list(state_dict.keys())[0]:
    state_dict = {k[7:]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mymodel = model.to(device)

theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="zinc",
).set(
    background_fill_primary='*neutral_100',
    background_fill_primary_dark='*neutral_900',
    background_fill_secondary='*neutral_200',
    border_color_accent='*secondary_500',
    border_color_accent_dark='*secondary_500',
    border_color_accent_subdued_dark='*secondary_500',
    border_color_primary='*neutral_400',
    border_color_primary_dark='*primary_950',
    color_accent_soft='*neutral_200',
    block_label_background_fill='*primary_200',
    block_label_border_color='*primary_300',
    block_label_radius='*radius_lg',
    block_title_text_color='*secondary_950',
    block_title_text_color_dark='*neutral_50',
    button_large_radius='*radius_xl',
    button_small_radius='*radius_xl',
    button_primary_background_fill='*secondary_400',
    button_primary_background_fill_dark='*secondary_800',
    button_primary_background_fill_hover='*secondary_500',
    button_secondary_background_fill='*secondary_400',
    button_secondary_background_fill_dark='*secondary_800',
    button_secondary_background_fill_hover='*secondary_500',
    button_secondary_background_fill_hover_dark='*secondary_400'
)

def numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

def process_image(image):
    image = preprocess(image).unsqueeze(0)
    image = extract_image_features_inst(image)
    return image.unsqueeze(0)

def process_question(question):
    question = question.lower()
    question = question.translate(str.maketrans('', '', string.punctuation))
    print(question)
    question = extract_text_features(question)
    return question.unsqueeze(0)

def make_prediction(image, question):
    image = process_image(image)
    question = process_question(question)
    output = mymodel(image, question)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    with open("label_to_answer_mapping.pkl", 'rb') as f:
        label_to_answers = pickle.load(f)
    
    top5_classes = [label_to_answers[catid.item()] for catid in top5_catid[0]]
    top5_probabilities = top5_prob[0].tolist()
    
    _, predicted = torch.max(output, 1)
    predicted_answer = label_to_answers[predicted.item()]
    
    return predicted_answer, top5_classes, top5_probabilities

def predict(image, question):
    if (image is None) and (question is None or question == ""):
        gr.Warning("Please upload an image and enter a question.")
        return
    if image is None:
        gr.Warning("Please upload an image.")
        return
    elif question is None or question == "":
        gr.Warning("Please what's your question?")
        return
    elif numeric(question) is True:
        gr.Warning("Please enter a valid question.")
        return
    else:
        gr.Info("Processing... Please Note that this may take a while. Thank you for your patience.")
        answer, top5_classes, top5_probabilities = make_prediction(image, question)
        progress_bars_html = "<div style='width: 50%; margin: auto;'>"
        for class_name, probability in zip(top5_classes, top5_probabilities):
            progress_bars_html += f"""
            <div style='margin-bottom: 8px;'>
                <span>{class_name}</span>
                <div style='background-color: #f3f3f3; border-radius: 4px; overflow: hidden;'>
                    <div style='width: {probability*100}%; background-color: #0284c7; padding: 8px 0; text-align: center; color: black; '>
                        {probability*100:.2f}%
                    </div>
                </div>
            </div>
            """
        progress_bars_html += "</div>"

        return answer, progress_bars_html
def clear_inputs():
    return None, "", ""
blocks = gr.Blocks(theme=theme,
                   css="""
        .header {
            text-align: center;
            padding: 0px;
            background-color: #f2f2f2;
        }
        .header img {
            max-width: 100%;
            max-height: 100%;
            margin-bottom: 0px;
        }
    """, title="MED-VQA")

with blocks as demo:
    gr.HTML("""<div class='header'><img src='http://localhost:8000/header.png' alt='Header Image'></div>""")
    with gr.Row():
        image = gr.Image(type="pil", label="Upload Image")
        question = gr.Textbox(lines=2, label="Question")
    answer = gr.Textbox(label="Answer")
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear")
    with gr.Accordion("See details"):
        chart = gr.HTML(label="Top 5 Probabilities")
    submit_button.click(fn=predict, inputs=[image, question], outputs=[answer, chart])
    clear_button.click(fn=clear_inputs, inputs=[], outputs=[image, question, answer])
# Launch the interface
demo.launch()
