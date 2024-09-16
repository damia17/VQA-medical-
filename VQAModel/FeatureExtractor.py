import torch
from torchvision import models
import torch
import torchvision.models as models
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
import numpy as np
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

#device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# load ViT extractor weights
pretrained_vit_weights = models.ViT_B_16_Weights.DEFAULT
pretrained_vit = models.vit_b_16(weights=pretrained_vit_weights).to(device)
# feature extraction using ViT
def image_features_extraction(model, input_tensor):
    cls_token_output = None

    def hook(module, input, output):
        nonlocal cls_token_output
        cls_token_output = output[:, 0] 

    handle = model.encoder.layers[-1].register_forward_hook(hook)
    with torch.no_grad():
        model(input_tensor)
    handle.remove()
    return cls_token_output
#   BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
text_encoder = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)

for p in text_encoder.parameters():
    p.requires_grad = False
# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def preprocess_image(img_id,type):
    if type == "train":
        img_path = "../datasets/clef2019/train/Train_images/" + img_id + ".jpg"
    elif type == "test":
        img_path = "../datasets/clef2019/test/Test_images/" + img_id + ".jpg"
    else:
        img_path = "../datasets/clef2019/valid/Val_images/" + img_id + ".jpg"
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0) #?[1,3,224,224]
# extraction image features (Preprocessing + ViT extractor)
def extract_image_features(img_id,type):
    pixel_values = preprocess_image(img_id,type).to(device) #?[1,3,224,224]
    outputs = image_features_extraction(pretrained_vit,pixel_values) #?[1,768]
    outputs = outputs.squeeze(0) #?[768]
    return outputs
# Extract image features for an input image tensor (For the GUI)
def extract_image_features_inst(img):
    outputs = image_features_extraction(pretrained_vit,img) #?[1,768]
    outputs = outputs.squeeze(0) #?[768]
    return outputs
# Extract question features avec BioBERT
def extract_text_features(text):
    text_inputs = tokenizer(text, return_tensors="pt").to(device) 
    text_inputs = {k:v for k,v in text_inputs.items()}
    text_outputs = text_encoder(**text_inputs)
    text_embedding = text_outputs.pooler_output 
    text_embedding = text_embedding.detach()
    text_embedding = text_embedding.squeeze(0) #?[768]
    return text_embedding
# Extract text features with Word2Vec
vector_size = 768
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
modelwv = FastText.load("wordtovec.model")
def get_sentence_embedding(sentence):
    tokens = sentence.lower().split()   
    tokens = [word for word in tokens if word not in stop_words]  
    word_vectors = [modelwv.wv[word] for word in tokens if word in modelwv.wv]
    
    if len(word_vectors) == 0:  
        return torch.zeros(768)
    sentence_embedding = np.mean(word_vectors, axis=0)
    
    sentence_embedding_tensor = torch.tensor(sentence_embedding) #?[768]
    return sentence_embedding_tensor 
