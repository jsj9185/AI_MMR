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


class MultiLSTM(nn.Module): # Bi-LSTM Model
    def __init__(self, config, mode):
        super(MultiLSTM, self).__init__()


        return None


class MultiBERT(nn.Module): # BERT based Model
    def __init__(self, config, mode):
        super(MultiBERT, self).__init__()


        return None
    
class MultiModalModel(nn.Module):
    """
    PyTorch implementation of the Polyvore Model for fashion compatibility tasks.
    """

    def __init__(self, config, mode="train", model_type="BiLSTM"):
        """
        Initialize the model.

        Args:
            config: Configuration object containing parameters.
            mode: "train", "eval", or "inference".
            model_type: Model type ("BiLSTM", "Transformer", or "BERT").
        """
        super(MultiModalModel, self).__init__()
        assert mode in ["train", "eval", "inference"], "Invalid mode"
        self.config = config
        self.mode = mode
        self.model_type = model_type

        # Model components
        self.image_embedder = None
        self.seq_embedder = None
        self.loss_fn = None


    def preprocess(self, image_features, text_features=None):

        return 0

    def image_embedding(self, image_features):

        return 

    def text_embedding(self, input_sequencese):

        return None

    def meta_embeding(self):

        return 0
    

    def build_model(self, image_features, input_sequences):
        
        return 0
    
    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return optimizer, scheduler

    def forward(self, image_features, input_sequences, attention_mask=None):
        self.preprocess()
        self.image_embedding()
        self.text_embedding()
        self.meta_embeding()
        self.build_model()
        self.train_model()
