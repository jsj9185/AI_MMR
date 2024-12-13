import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PolyvoreLSTMModel(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        hidden_dim=512,
        num_layers=None,
        dropout=0.5,
        contrastive_loss_factor=1.0,
        mode='train'  # 'train' 또는 'inference'
    ):
        super(PolyvoreLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.contrastive_loss_factor = contrastive_loss_factor
        self.mode = mode

        # 단방향 LSTM
        self.forward_lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        # 단방향 LSTM
        self.backward_lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        # rnn_output_dim = hidden_dim * 2
        self.fc = nn.Linear(embedding_size*2, hidden_dim)
        # self.emb_margin = 0.2

    '''def forward(self, image_embeddings, text_embeddings, lengths, mask=None):
        sliced_image_embeddings = image_embeddings[:, :-1, :]  # 1~k-1
        sliced_text_embeddings = text_embeddings[:, :-1, :]  # 1~k-1
        sliced_lengths = lengths - 1
        # Forward 
        forward_out = self._run_lstm(sliced_image_embeddings, sliced_text_embeddings, sliced_lengths, direction='forward')
        prediction = forward_out[:, -1, :]
        gt = self.combine_embeddings(image_embeddings[:, -1:, :], text_embeddings[:, -1:, :]).squeeze(1)
        print(prediction.shape, gt.shape)
        # Backward Pass
        rev_image = self._reverse_seq(image_embeddings, lengths)
        rev_text = self._reverse_seq(text_embeddings, lengths)
        backward_out = self._run_lstm(rev_image, rev_text, lengths, direction='backward')
        backward_out = self._reverse_seq(backward_out, lengths)
        #print("outshape")
        #print(forward_out.shape, backward_out.shape)

        if self.mode == 'train':
            # Combine Forward and Backward outputs for Training
            #combined_out = torch.cat([forward_out, backward_out], dim=2)
            #rnn_output_embeddings = F.normalize(self.fc(combined_out), p=2, dim=2)
            #combined_out = forward_out + backward_out
            #rnn_output_embeddings = F.normalize(combined_out, p=2, dim=2)

            assert mask is not None, "Train mode requires mask"
            #total_loss = self.compute_total_loss(
            #    rnn_output_embeddings=rnn_output_embeddings,
            #    mask=mask)
            total_loss = self.compute_total_loss(
            forward_out=forward_out, backward_out=backward_out,
            mask=mask)
            return total_loss
        else:
            return forward_out, backward_out'''

    def forward(self, image_embeddings, text_embeddings, lengths, mask=None, mode='direct'):
        """
        Forward 메서드 수정: Forward와 Backward를 모두 고려.
        - Forward: 1~k-1을 입력으로 받아 k_hat 예측.
        - Backward: k~2를 입력으로 받아 1_hat 예측.
        """
        # Forward Pass (1~k-1)
        sliced_image_embeddings = image_embeddings[:, :-1, :]  # 1~k-1
        sliced_text_embeddings = text_embeddings[:, :-1, :]  # 1~k-1
        sliced_lengths = lengths - 1  # 각 시퀀스 길이에서 1씩 감소


        forward_out = self._run_lstm(
            sliced_image_embeddings, sliced_text_embeddings, sliced_lengths, direction='forward'
        )  # Shape: [B, T-1, H]

        k_hat = forward_out[:, -1, :]  # Forward에서 예측된 마지막 값 (k_hat)
        actual_k = self.combine_embeddings(
            image_embeddings[:, -1:, :], text_embeddings[:, -1:, :]
        ).squeeze(1)  # 실제 k 값

        # Backward Pass (k~2)
        rev_image_embeddings = self._reverse_seq(image_embeddings, lengths)
        rev_text_embeddings = self._reverse_seq(text_embeddings, lengths)
        backward_out = self._run_lstm(
            rev_image_embeddings[:, :-1, :], rev_text_embeddings[:, :-1, :], sliced_lengths, direction='backward'
        )  # Shape: [B, T-1, H]

        reversed_1_hat = backward_out[:, -1, :]  # Backward에서 예측된 마지막 값 (1_hat)
        actual_1 = self.combine_embeddings(
            image_embeddings[:, :1, :], text_embeddings[:, :1, :]
        ).squeeze(1)  # 실제 1 값

        # 학습 또는 추론 모드에 따른 처리
        if self.mode == 'train':
            # Forward 손실 계산: k_hat vs actual_k (배치 내 비교)
            forward_loss = self.compute_batch_loss(k_hat, actual_k)

            # Backward 손실 계산: 1_hat vs actual_1 (배치 내 비교)
            backward_loss = self.compute_batch_loss(reversed_1_hat, actual_1)

            # 총 손실 계산
            total_loss = forward_loss + backward_loss
            return total_loss
        else:
            # 추론 모드
            return k_hat, reversed_1_hat

    def compute_batch_loss(self, k_hat, actual_k):
        """
        배치 내 모든 후보들과 비교하여 크로스엔트로피 손실 계산.
        """
        # 배치 내 모든 점곱 계산: [B, H] x [H, B] -> [B, B]
        k_hat = F.normalize(k_hat, p=2, dim=-1)
        actual_k = F.normalize(actual_k, p=2, dim=-1)   
        scores = torch.matmul(k_hat, actual_k.T)  # 유사도 행렬
        #print(scores.shape)
        targets = torch.arange(scores.size(0), device=scores.device)  # 정답 인덱스 (대각선)
        #print(targets)

        # 크로스엔트로피 손실 계산
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, targets)
        return loss

    def compute_total_loss(self, forward_out, backward_out, mask):
        # Forward/Backward Loss 계산 및 정규화
        #forward_loss, backward_loss = self.compute_rnn_loss(rnn_output_embeddings, mask)
        forward_loss = self.compute_rnn_loss(forward_out, mask)
        backward_loss = self.compute_rnn_loss(backward_out, mask)
        
        # 총 손실 계산
        total_loss = forward_loss + backward_loss

        return total_loss

    def compute_rnn_loss(self, rnn_output_embeddings, mask):
        forward_input = rnn_output_embeddings[:, :-1, :]
        forward_target = rnn_output_embeddings[:, 1:, :]
        forward_mask = mask[:, :-1].bool() & mask[:, 1:].bool()  # boolean 변환
        rnn_loss = self._compute_single_rnn_loss(forward_input, forward_target, forward_mask)

        return rnn_loss

    def _compute_single_rnn_loss(self, input_embeddings, target_embeddings, mask):
        input_valid = input_embeddings[mask.bool()]
        target_valid = target_embeddings[mask.bool()]
        #print('validshape')
        #print(input_valid.shape, target_valid.shape)
        scores = torch.matmul(input_valid, target_valid.T)
        #print(scores.shape)
        targets = torch.arange(scores.size(0), device=scores.device)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(scores, targets)
    
    def combine_embeddings(self, image_embeddings, text_embeddings):
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=2)  # [batch_size, seq_len, 1024]
        return self.fc(combined_embeddings)

    def _run_lstm(self, image_embeddings, text_embeddings, lengths, direction='forward'):
        combined_embeddings = self.combine_embeddings(image_embeddings, text_embeddings)
        #combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=2)
        #print(combined_embeddings.shape)
        packed_inputs = pack_padded_sequence(
            combined_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if direction == 'forward':
            lstm_model = self.forward_lstm
        else:
            lstm_model = self.backward_lstm
        packed_outputs, _ = lstm_model(packed_inputs)
        lstm_out, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=combined_embeddings.size(1))
        return lstm_out

    def _reverse_seq(self, x, lengths):
        bsz, seq_len, dim = x.size()
        x_reversed = torch.zeros_like(x)
        for i in range(bsz):
            l = lengths[i].item()
            x_reversed[i, :l] = x[i, :l].flip(dims=[0])
        return x_reversed


'''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PolyvoreLSTMModel(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        hidden_dim=512,
        num_layers=1,
        dropout=0.5,
        mode='train'  # 'train' or 'eval'
    ):
        super(PolyvoreLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.mode = mode

        self.forward_lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        self.backward_lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        # fc 레이어를 통해 image/text embedding -> hidden_dim 차원으로 매핑
        self.fc = nn.Linear(embedding_size*2, hidden_dim)

    def forward(self, image_embeddings, text_embeddings, lengths):
        if self.mode == 'train':
            # Train 모드에서는 전체 시퀀스를 사용
            sliced_image_embeddings = image_embeddings[:, :-1, :]  # 1~k-1
            sliced_text_embeddings = text_embeddings[:, :-1, :]  # 1~k-1
            sliced_lengths = lengths - 1
        else:  # inference 모드
            # Inference 모드에서는 이미 잘린 시퀀스를 사용 (추가 슬라이싱 불필요)
            sliced_image_embeddings = image_embeddings  # 그대로 사용
            sliced_text_embeddings = text_embeddings  # 그대로 사용
            sliced_lengths = lengths  # 그대로 사용

        print(image_embeddings.shape, sliced_image_embeddings.shape)
        forward_out = self._run_lstm(
            sliced_image_embeddings, sliced_text_embeddings, sliced_lengths, direction='forward'
        )  # (B, k-1, H)
        k_hat = forward_out[:, -1, :]  # (B, H)
        actual_k = self.combine_embeddings(
            image_embeddings[:, -1:, :], text_embeddings[:, -1:, :]
        ).squeeze(1)  # (B, H)

        # Backward Pass (k~2)
        rev_image_embeddings = self._reverse_seq(image_embeddings, lengths)
        rev_text_embeddings = self._reverse_seq(text_embeddings, lengths)
        backward_out = self._run_lstm(
            rev_image_embeddings[:, :-1, :], rev_text_embeddings[:, :-1, :], sliced_lengths, direction='backward'
        )  # (B, k-1, H)
        reversed_1_hat = backward_out[:, -1, :]  # (B, H)
        actual_1 = self.combine_embeddings(
            image_embeddings[:, :1, :], text_embeddings[:, :1, :]
        ).squeeze(1)  # (B, H)

        if self.mode == 'train':
            # 단순 MSE loss로 임베딩 회귀
            forward_loss = F.mse_loss(k_hat, actual_k)
            backward_loss = F.mse_loss(reversed_1_hat, actual_1)
            total_loss = forward_loss + backward_loss
            return total_loss
        else:
            # inference 모드에서는 예측값만 반환
            return k_hat, reversed_1_hat

    def combine_embeddings(self, image_embeddings, text_embeddings):
        # image와 text 임베딩을 concat후 fc로 매핑 -> (B, T, H)
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=2)
        return self.fc(combined_embeddings)

    def _run_lstm(self, image_embeddings, text_embeddings, lengths, direction='forward'):
        #print(image_embeddings.shape, text_embeddings.shape)
        combined_embeddings = self.combine_embeddings(image_embeddings, text_embeddings)
        #print(combined_embeddings.shape)
        packed_inputs = pack_padded_sequence(
            combined_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_model = self.forward_lstm if direction == 'forward' else self.backward_lstm
        packed_outputs, _ = lstm_model(packed_inputs)
        lstm_out, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=combined_embeddings.size(1))
        return lstm_out

    def _reverse_seq(self, x, lengths):
        bsz, seq_len, dim = x.size()
        x_reversed = torch.zeros_like(x)
        for i in range(bsz):
            l = lengths[i].item()
            x_reversed[i, :l] = x[i, :l].flip(dims=[0])
        return x_reversed
'''