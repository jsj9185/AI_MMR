import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from PIL import Image
from importlib import reload
from dataloader.sampling import DataSampler  # 이미 import된 모듈
from importlib import reload
from dataloader.multimodal_data import MultiModalData
import os
from torch.utils.data import Dataset, DataLoader
from model.multifusion import Multifusion
import torch.nn.functional as F

path = 'data/meta/'
train = pd.read_json(path + 'train_no_dup.json')
valid = pd.read_json(path + 'valid_no_dup.json')
test = pd.read_json(path + 'test_no_dup.json')
blank = pd.read_json(path + 'fill_in_blank_test.json')


base_dir = ''
data_dir = os.path.join(base_dir, 'data')
meta_dir = os.path.join(data_dir, 'meta')
image_dir = os.path.join(data_dir, 'images')
sampler = DataSampler(data_path = meta_dir,  test_sampling_ratio=0.33)
concat_df, question_data = sampler.sample_data()


train_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='train')
valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')


device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
model = Multifusion().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def cosine_similarity_loss(image_context, image_gt, text_context, text_gt):
    # 이미지 임베딩의 Cosine Similarity 계산 및 손실 정의
    image_similarity = F.cosine_similarity(image_context, image_gt, dim=-1)
    loss_image = 1 - image_similarity.mean()  # 유사도를 최대화
    
    # 텍스트 임베딩의 Cosine Similarity 계산 및 손실 정의
    text_similarity = F.cosine_similarity(text_context, text_gt, dim=-1)
    loss_text = 1 - text_similarity.mean()  # 유사도를 최대화

    # 최종 손실 합산
    total_loss = loss_image + loss_text
    return total_loss


num_epochs = 50
# 학습 루프
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # 데이터 로드
        image_embedding = batch['images'] # Shape: (B, 7, 512)
        text_embedding = batch['texts']  # Shape: (B, 7, 512)
        image_gt = batch['image_gt']    # Shape: (B, 1, 512)
        text_gt = batch['text_gt']      # Shape: (B, 1, 512)

        # Forward pass
        output = model(image_embedding, text_embedding)  # Shape: (B, 2, 512)

        # Output 분리
        image_context = output[:, 0, :]  # Shape: (B, 512)
        text_context = output[:, 1, :]   # Shape: (B, 512)

        # 손실 계산
        loss = cosine_similarity_loss(image_context, image_gt, text_context, text_gt)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 로그 출력
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")
            
            
torch.save(model.state_dict(), "multifusion_model_nosampling_epoch50.pth")
print("Model saved to multifusion_model.pth")