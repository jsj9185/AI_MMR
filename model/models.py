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




import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLSTM(nn.Module):  # Bi-LSTM Model
    def __init__(self, input_size=512, hidden_size=512, num_layers=1, bidirectional=True):
        super(MultiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.image_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.text_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.image_fc = nn.Linear(lstm_output_size, input_size)
        self.text_fc = nn.Linear(lstm_output_size, input_size)

    def forward(self, image_embedding, text_embedding, gt_idx):
        batch_size, seq_len, embedding_size = image_embedding.size()
        device = image_embedding.device

        seq_indices = torch.arange(seq_len).to(device).unsqueeze(0).expand(batch_size, -1)

        mask = seq_indices != gt_idx.unsqueeze(1)

        image_seq = image_embedding[mask].view(batch_size, seq_len - 1, embedding_size)
        text_seq = text_embedding[mask].view(batch_size, seq_len - 1, embedding_size)

        image_output, (h_n_image, _) = self.image_lstm(image_seq)
        text_output, (h_n_text, _) = self.text_lstm(text_seq)

        if self.bidirectional:
            h_n_image_forward = h_n_image[-2, :, :]
            h_n_image_backward = h_n_image[-1, :, :]
            h_n_image_cat = torch.cat((h_n_image_forward, h_n_image_backward), dim=1)
            
            h_n_text_forward = h_n_text[-2, :, :]
            h_n_text_backward = h_n_text[-1, :, :]
            h_n_text_cat = torch.cat((h_n_text_forward, h_n_text_backward), dim=1)
        else:
            h_n_image_cat = h_n_image[-1, :, :]
            h_n_text_cat = h_n_text[-1, :, :]

        image_context = self.image_fc(h_n_image_cat)
        text_context = self.text_fc(h_n_text_cat)


        return image_context, text_context
'''
model = MultiLSTM(input_size=512, hidden_size=512, num_layers=1, bidirectional=True)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def cosine_similarity_loss(image_context, image_gt, text_context, text_gt):
    image_loss = 1 - F.cosine_similarity(image_context, image_gt, dim=-1).mean()
    text_loss = 1 - F.cosine_similarity(text_context, text_gt, dim=-1).mean()
    total_loss = (image_loss + text_loss) / 2
    return total_loss

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        image_embedding = batch['images'].to(device)
        text_embedding = batch['texts'].to(device)
        gt_idx = batch['gt_idx'].to(device)
        batch_size = image_embedding.size(0)

        image_gt = image_embedding[torch.arange(batch_size), gt_idx, :]
        text_gt = text_embedding[torch.arange(batch_size), gt_idx, :]

        image_context, text_context = model(image_embedding, text_embedding, gt_idx)

        loss = cosine_similarity_loss(image_context, image_gt, text_context, text_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "multilstm_model_epoch50.pth")
print("Model saved to multilstm_model_epoch50.pth")
'''

class MultiTransformer(nn.Module):
    def __init__(self, embedding_dim=512, num_layers=4, num_heads=8, dropout=0.1, max_seq_len=14):
        super(MultiTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, image_embedding, text_embedding, gt_idx):
        batch_size, seq_len, embedding_dim = image_embedding.size()
        device = image_embedding.device

        embeddings = torch.cat([image_embedding, text_embedding], dim=1) # embedding concat
        seq_len = embeddings.size(1)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        seq_indices = torch.arange(seq_len).to(device).unsqueeze(0).expand(batch_size, -1)
        mask = seq_indices != gt_idx.unsqueeze(1)

        masked_embeddings = embeddings.clone()
        masked_embeddings[~mask] = 0

        transformer_input = masked_embeddings.transpose(0, 1)

        padding_mask = ~mask

        transformer_output = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=padding_mask
        )

        transformer_output = transformer_output.transpose(0, 1)

        predicted_embeddings = transformer_output[torch.arange(batch_size), gt_idx, :]

        output = self.output_layer(predicted_embeddings)

        return output
    
"""
model = MultiBERT(embedding_dim=512, num_layers=4, num_heads=8, dropout=0.1, max_seq_len=14)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def cosine_similarity_loss(predicted_embedding, gt_embedding):
    loss = 1 - F.cosine_similarity(predicted_embedding, gt_embedding, dim=-1).mean()
    return loss

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        image_embedding = batch['images'].to(device)
        text_embedding = batch['texts'].to(device)
        gt_idx = batch['gt_idx'].to(device)
        batch_size = image_embedding.size(0)

        embeddings = torch.cat([image_embedding, text_embedding], dim=1)

        gt_idx = gt_idx + 7

        gt_embedding = text_embedding[torch.arange(batch_size), gt_idx - 7, :]

        predicted_embedding = model(image_embedding, text_embedding, gt_idx)

        loss = cosine_similarity_loss(predicted_embedding, gt_embedding)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "multibert_model_epoch50.pth")
print("Model saved to multibert_model_epoch50.pth")
"""




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
