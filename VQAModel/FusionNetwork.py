import torch
import torch.nn as nn
#Fusion Network based on multi-head attention 
class MultiHeadAttentionFusionNetwork(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4):
        super(MultiHeadAttentionFusionNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        self.W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.relu = nn.ReLU()

    def forward(self, image_emb, text_emb):
        # normalisation
        x1 = torch.nn.functional.normalize(image_emb, p=2, dim=1)  #?[batch_size,768]
        x2 = torch.nn.functional.normalize(text_emb, p=2, dim=1)  #?[batch_size,768]
        x1 = x1.unsqueeze(0)  #?[1,batch_size,768]
        x2 = x2.unsqueeze(0)  #?[1,batch_size,768]
        # attention 
        attn_output, _ = self.multihead_attn(x1, x2, x2)  

        attn_output = attn_output.squeeze(0)  #?[batch_size,768]

        Xvt = image_emb * attn_output  
        # multiplication 
        fused_representation = torch.matmul(Xvt, self.W.t())  

        fused_representation = self.layer_norm(fused_representation)
        
        return fused_representation #?[batch_size,768]
