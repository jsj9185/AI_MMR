import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyvoreLSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_dim, num_layers=1, bidirectional=True, dropout=0.5):
        super(PolyvoreLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully Connected Layers for mapping to embedding space and RNN prediction space
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.image_embedding_fc = nn.Linear(embedding_size, embedding_size)
        self.rnn_output_fc = nn.Linear(rnn_output_dim, embedding_size)

        # Parameters for the contrastive loss
        self.emb_margin = 0.2  # Margin for the contrastive loss

    def forward(self, image_embeddings, lengths):
        """
        Forward pass through the model.

        Args:
        - image_embeddings (torch.Tensor): Tensor of shape [batch_size, sequence_length, embedding_size]
        - lengths (torch.Tensor): Tensor containing the lengths of each sequence in the batch

        Returns:
        - rnn_output_embeddings (torch.Tensor): Output embeddings from the RNN, shape [batch_size, sequence_length, embedding_size]
        - image_embeddings_transformed (torch.Tensor): Transformed image embeddings, shape [batch_size, sequence_length, embedding_size]
        """
        batch_size, seq_len, embed_size = image_embeddings.size()

        # Transform image embeddings to RNN input space
        rnn_inputs = self.image_embedding_fc(image_embeddings)

        # Pack the sequences for efficient processing
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            rnn_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through the Bi-LSTM
        packed_outputs, _ = self.lstm(packed_inputs)

        # Unpack the sequences
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=seq_len)

        # Transform RNN outputs back to embedding space
        rnn_output_embeddings = self.rnn_output_fc(lstm_out)

        # Normalize embeddings
        image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=2)
        rnn_output_embeddings_normalized = F.normalize(rnn_output_embeddings, p=2, dim=2)

        return rnn_output_embeddings_normalized, image_embeddings_normalized

    def compute_contrastive_loss(self, image_embeddings, seq_embeddings, mask):
        """
        Computes the contrastive loss between image and sequence embeddings.

        Args:
        - image_embeddings (torch.Tensor): Normalized image embeddings, shape [batch_size, sequence_length, embedding_size]
        - seq_embeddings (torch.Tensor): Normalized sequence embeddings, shape [batch_size, sequence_length, embedding_size]
        - mask (torch.Tensor): Binary mask indicating valid positions, shape [batch_size, sequence_length]

        Returns:
        - loss (torch.Tensor): Scalar tensor representing the contrastive loss
        """
        batch_size, seq_len, embed_size = image_embeddings.size()

        # Flatten the embeddings and mask
        image_embeddings_flat = image_embeddings.view(-1, embed_size)
        seq_embeddings_flat = seq_embeddings.view(-1, embed_size)
        mask_flat = mask.view(-1).bool()

        # Apply mask
        image_embeddings_valid = image_embeddings_flat[mask_flat]
        seq_embeddings_valid = seq_embeddings_flat[mask_flat]

        # Compute scores
        scores = torch.matmul(seq_embeddings_valid, image_embeddings_valid.T)

        # Compute diagonal (positive scores)
        diag = scores.diag().unsqueeze(1)

        # Compute contrastive loss
        cost_s = F.relu(self.emb_margin - diag + scores)
        cost_im = F.relu(self.emb_margin - diag.T + scores)
        # Zero out the diagonal terms
        mask = torch.eye(cost_s.size(0), device=cost_s.device).bool()
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        loss = (cost_s.sum() + cost_im.sum()) / (cost_s.size(0) ** 2)
        return loss

    def compute_rnn_loss(self, rnn_output_embeddings, target_embeddings, mask):
        """
        Computes the loss between RNN outputs and target embeddings.

        Args:
        - rnn_output_embeddings (torch.Tensor): Output from the RNN, shape [batch_size * sequence_length, embedding_size]
        - target_embeddings (torch.Tensor): Target embeddings, shape [batch_size * sequence_length, embedding_size]
        - mask (torch.Tensor): Binary mask indicating valid positions, shape [batch_size * sequence_length]

        Returns:
        - loss (torch.Tensor): Scalar tensor representing the RNN loss
        """
        # Apply mask
        rnn_output_valid = rnn_output_embeddings[mask]
        target_valid = target_embeddings[mask]

        # Compute scores
        scores = torch.matmul(rnn_output_valid, target_valid.T)

        # Compute CrossEntropyLoss over batch
        targets = torch.arange(scores.size(0), device=scores.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, targets)

        return loss

    def compute_total_loss(self, image_embeddings, seq_embeddings, rnn_output_embeddings, target_embeddings, mask):
        """
        Computes the total loss combining contrastive and RNN losses.

        Args:
        - image_embeddings (torch.Tensor): Original image embeddings
        - seq_embeddings (torch.Tensor): Corresponding sequence embeddings
        - rnn_output_embeddings (torch.Tensor): Output from the RNN
        - target_embeddings (torch.Tensor): Target embeddings for the RNN
        - mask (torch.Tensor): Binary mask indicating valid positions

        Returns:
        - total_loss (torch.Tensor): Scalar tensor representing the total loss
        """
        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(image_embeddings, seq_embeddings, mask)

        # Flatten embeddings for RNN loss computation
        batch_size, seq_len, embed_size = rnn_output_embeddings.size()
        rnn_output_flat = rnn_output_embeddings.view(-1, embed_size)
        target_flat = target_embeddings.view(-1, embed_size)
        mask_flat = mask.view(-1).bool()

        # Compute RNN loss (forward and backward)
        rnn_loss = self.compute_rnn_loss(rnn_output_flat, target_flat, mask_flat)

        # Total loss (weighted sum)
        total_loss = contrastive_loss + rnn_loss

        return total_loss


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 필요한 모듈들 임포트
from dataset import PolyvoreDataset  # 사용자 정의 데이터셋 클래스
from model import PolyvoreLSTMModel  # 이전에 정의한 모델 클래스
from torch.nn.utils.rnn import pad_sequence

# 학습 설정 클래스 정의 (옵션에 따라 수정 가능)
class Config:
    embedding_size = 512  # CLIP 임베딩 크기
    hidden_dim = 256
    num_layers = 1
    bidirectional = True
    dropout = 0.5
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 32

# 데이터셋 및 데이터로더 준비
def collate_fn(batch):
    """
    배치 내의 샘플들을 패딩하고 필요한 형태로 변환합니다.
    """
    image_embeddings = [item['image_embeddings'] for item in batch]
    seq_embeddings = [item['seq_embeddings'] for item in batch]
    lengths = torch.tensor([len(seq) for seq in image_embeddings])

    # 패딩
    image_embeddings_padded = pad_sequence(image_embeddings, batch_first=True)
    seq_embeddings_padded = pad_sequence(seq_embeddings, batch_first=True)
    mask = (image_embeddings_padded.sum(dim=2) != 0).float()  # 패딩이 아닌 부분 마스크

    return {
        'image_embeddings': image_embeddings_padded,
        'seq_embeddings': seq_embeddings_padded,
        'lengths': lengths,
        'mask': mask
    }

# 학습 함수 정의
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # 데이터 로드
        image_embeddings = batch['image_embeddings'].to(device)
        seq_embeddings = batch['seq_embeddings'].to(device)
        lengths = batch['lengths'].to(device)
        mask = batch['mask'].to(device)

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 모델 예측
        rnn_output_embeddings, image_embeddings_normalized = model(image_embeddings, lengths)

        # 텍스트 임베딩 정규화
        seq_embeddings_normalized = F.normalize(seq_embeddings, p=2, dim=2)

        # 타겟 임베딩 생성 (시퀀스 한 칸씩 시프트)
        target_embeddings = torch.zeros_like(rnn_output_embeddings)
        for i in range(rnn_output_embeddings.size(0)):
            target_embeddings[i, :-1, :] = image_embeddings_normalized[i, 1:, :]
            target_embeddings[i, -1, :] = torch.zeros(model.embedding_size).to(device)

        # 총 손실 계산
        loss = model.compute_total_loss(
            image_embeddings=image_embeddings_normalized,
            seq_embeddings=seq_embeddings_normalized,
            rnn_output_embeddings=rnn_output_embeddings,
            target_embeddings=target_embeddings,
            mask=mask
        )

        # 역전파 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

# 검증 함수 정의
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # 데이터 로드
            image_embeddings = batch['image_embeddings'].to(device)
            seq_embeddings = batch['seq_embeddings'].to(device)
            lengths = batch['lengths'].to(device)
            mask = batch['mask'].to(device)

            # 모델 예측
            rnn_output_embeddings, image_embeddings_normalized = model(image_embeddings, lengths)

            # 텍스트 임베딩 정규화
            seq_embeddings_normalized = F.normalize(seq_embeddings, p=2, dim=2)

            # 타겟 임베딩 생성 (시퀀스 한 칸씩 시프트)
            target_embeddings = torch.zeros_like(rnn_output_embeddings)
            for i in range(rnn_output_embeddings.size(0)):
                target_embeddings[i, :-1, :] = image_embeddings_normalized[i, 1:, :]
                target_embeddings[i, -1, :] = torch.zeros(model.embedding_size).to(device)

            # 총 손실 계산
            loss = model.compute_total_loss(
                image_embeddings=image_embeddings_normalized,
                seq_embeddings=seq_embeddings_normalized,
                rnn_output_embeddings=rnn_output_embeddings,
                target_embeddings=target_embeddings,
                mask=mask
            )

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

# 메인 함수 정의
def main():
    # 설정 로드
    config = Config()

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 초기화
    model = PolyvoreLSTMModel(
        embedding_size=config.embedding_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout
    ).to(device)

    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 데이터셋 및 데이터로더 설정
    train_dataset = PolyvoreDataset(split='train')  # 사용자 정의 데이터셋 클래스 사용
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 검증 데이터셋 및 데이터로더 설정
    valid_dataset = PolyvoreDataset(split='valid')  # 'valid' 스플릿 사용
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 학습 루프 시작
    for epoch in range(config.num_epochs):
        # 학습 단계
        average_loss = train(model, train_dataloader, optimizer, device)
        print(f'Epoch [{epoch + 1}/{config.num_epochs}], Training Loss: {average_loss:.4f}')

        # 검증 단계
        validation_loss = validate(model, valid_dataloader, device)
        print(f'Epoch [{epoch + 1}/{config.num_epochs}], Validation Loss: {validation_loss:.4f}')

        # 체크포인트 저장 (필요에 따라)
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()


import torch
from model import PolyvoreLSTMModel

def load_model(model_path, input_dim, hidden_dim, output_dim, num_layers, bidirectional, device):
    model = PolyvoreLSTMModel(input_dim, hidden_dim, output_dim, num_layers, bidirectional)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, inputs, device):
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs, _ = model(inputs)
    return outputs

def main():
    model_path = "best_model.pth"
    input_dim = 512
    hidden_dim = 256
    output_dim = 512
    num_layers = 1
    bidirectional = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, input_dim, hidden_dim, output_dim, num_layers, bidirectional, device)

    # Example input
    inputs = torch.randn(1, 8, input_dim)  # [batch_size, seq_length, input_dim]
    predictions = predict(model, inputs, device)
    print("Predictions:", predictions)