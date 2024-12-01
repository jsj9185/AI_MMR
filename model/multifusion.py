import torch
from torch import nn
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)
# image2 = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["a dog"]).to(device)
# prices, likes = 3000, 5
# meta = torch.tensor([prices, likes], dtype=torch.float32).unsqueeze(0).to(device)
# image => [1, 3, 224, 224]  => (CLIP feature extractor) => [1, 512] 
# text (word-wise token embedding) => [1, 77] => (CLIP feature extractor) => [1, 512]
# prices, likes => [1, 2] => (MLP) => [1, 512]
# fused feature [3, 512] = [[image feature], [text feature], [price & like feature]]

import torch
import torch.nn as nn
from transformers import BertModel

class Multifusion(nn.Module): # Basic MLP Model
  def __init__(self):
    super(Multifusion, self).__init__()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.ffn_image = nn.Linear(512 * 7, 512)
    self.ffn_text = nn.Linear(512 * 7, 512)
    
  def forward(self, image_embedding, text_embedding):
    print("model start")
    batch = image_embedding.shape[0]
    
    # image_embedding = torch.cat((image_embedding[:, : gt_idx, :] , image_embedding[:, gt_idx+1:, :]), dim=1)
    # text_embedding = torch.cat((text_embedding[:, : gt_idx, :] , text_embedding[:, gt_idx+1:, :]), dim=1)
    
    image_embedding = torch.flatten(image_embedding, 1)
    text_embedding = torch.flatten(text_embedding, 1)
    
    # 특징들을 concatenate
    image_context = self.ffn_image(image_embedding).unsqueeze(1) #[B, 1, 512]
    text_context = self.ffn_text(text_embedding).unsqueeze(1)    #[B, 1, 512]
    
    out = torch.cat((image_context, text_context), dim=1) #[B, 2, 512]
    print("model end")
    return out
