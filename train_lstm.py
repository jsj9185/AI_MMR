import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
from datetime import datetime
from dataloader.sampling import DataSampler
from dataloader.multilstm_data import MultiModalData
from model.multilstm import PolyvoreLSTMModel

import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

def collate_fn(batch):
    """
    배치 내 데이터를 패딩 처리하고 필요한 키로 구성된 dictionary 반환.
    """
    image_embeddings = [item['images'] for item in batch]
    seq_embeddings = [item['texts'] for item in batch]
    lengths = torch.tensor([len(seq) for seq in image_embeddings])

    image_embeddings_padded = pad_sequence(image_embeddings, batch_first=True)
    seq_embeddings_padded = pad_sequence(seq_embeddings, batch_first=True)
    mask = (image_embeddings_padded.sum(dim=2) != 0).float()

    return {
        'image_embeddings': image_embeddings_padded,
        'seq_embeddings': seq_embeddings_padded,
        'lengths': lengths,
        'mask': mask
    }


def train(model, dataloader, optimizer, device):
    """
    학습 루프 함수: 데이터를 사용해 모델을 학습.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        image_embeddings = batch['image_embeddings'].to(device)
        seq_embeddings = batch['seq_embeddings'].to(device)
        lengths = batch['lengths'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        # 모델 호출: 정규화 및 loss 계산 모두 모델 내부에서 처리
        loss = model(
            image_embeddings=image_embeddings,
            text_embeddings=seq_embeddings,
            lengths=lengths,
            #mask=mask
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    검증 루프 함수: 데이터를 사용해 모델의 성능을 평가.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            image_embeddings = batch['image_embeddings'].to(device)
            seq_embeddings = batch['seq_embeddings'].to(device)
            lengths = batch['lengths'].to(device)
            mask = batch['mask'].to(device)

            # 모델 호출: loss 계산
            loss = model(
                image_embeddings=image_embeddings,
                text_embeddings=seq_embeddings,
                lengths=lengths,
                #mask=mask
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)

class Config:
    embedding_size = 512
    hidden_dim = 512
    num_layers = 3
    dropout = 0.5
    learning_rate = 0.005
    num_epochs = 30
    batch_size = 16
    #contrastive_loss_factor = 1.0
    category_k = 150

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("################ Code Start ################\n")
    # 모델 초기화
    model = PolyvoreLSTMModel(
        embedding_size=config.embedding_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        #contrastive_loss_factor=config.contrastive_loss_factor,
        mode='train'
    ).to(device)
    print("################ Model Loaded ################\n")

    # 옵티마이저 초기화
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 데이터 로드 및 전처리
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    meta_dir = os.path.join(data_dir, 'meta')
    image_dir = os.path.join(data_dir, 'images')
    #print("Trial 4")

    sampler = DataSampler(data_path=meta_dir, k=config.category_k, test_sampling_ratio=1)
    concat_df, question_data = sampler.sample_data()

    train_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='train')
    valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
    print("################ Data Loaded ################\n")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    print("################ Train Start ################\n")

    # 학습 루프
    best_loss = 100.0
    current_time = datetime.now().strftime("%m%d%H%M")
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    checkpoint_dir = os.path.join(checkpoints_dir, f'checkpoint_{current_time}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Training Loss: {train_loss:.4f}")

        valid_loss = validate(model, valid_dataloader, device)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Validation Loss: {valid_loss:.4f}")

        # 모델 체크포인트 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                }, 
                os.path.join(checkpoint_dir, "best_checkpoint.pth")
            )
            print(f"Checkpoint saved to {checkpoint_dir}")

if __name__ == '__main__':
    main()
