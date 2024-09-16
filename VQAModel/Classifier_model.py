from torch import nn
import torch
from torchvision import models

# Classe de classification (Bassee sur l'encodeur-classificateur ViT)

class Classifier(nn.Module):
    def __init__(self,  num_classes, dropout_prob):
        super(Classifier, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.positional_encoding = nn.Parameter(self.vit.encoder.pos_embedding[:, :197, :])
        self.dropout = nn.Dropout(p=dropout_prob)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 196, 1)  
        batch_size = x.size(0)
        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  

        x = torch.cat((cls_token, x), dim=1)  
        #pass to the encoder
        x = self.vit.encoder(x)  
        x = self.vit.encoder.ln(x)  
        x = self.dropout(x)
        
        x = self.vit.heads(x[:, 0])  
        return x