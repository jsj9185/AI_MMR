import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PolyvoreLSTMModel(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        hidden_dim=256,
        num_layers=1,
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
        self.lstm = nn.LSTM(
            input_size=embedding_size * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 단방향 출력이지만 forward/backward 시퀀스 처리로 양방향 효과를 낼 계획이라 fc input_dim=hidden_dim*2
        rnn_output_dim = hidden_dim * 2
        self.fc = nn.Linear(rnn_output_dim, embedding_size)
        self.emb_margin = 0.2

    def forward(self, image_embeddings, text_embeddings, lengths, mask=None):
        # 원본 임베딩 정규화 (forward 내부로 이동)
        image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=2)
        text_embeddings_normalized = F.normalize(text_embeddings, p=2, dim=2)

        # forward pass
        forward_out = self._run_lstm_forward(image_embeddings, text_embeddings, lengths)

        if self.mode == 'train':
            # backward pass: 시퀀스를 reverse하여 LSTM 통과 후 다시 reverse
            rev_image = self._reverse_seq(image_embeddings, lengths)
            rev_text = self._reverse_seq(text_embeddings, lengths)
            backward_out = self._run_lstm_forward(rev_image, rev_text, lengths)
            backward_out = self._reverse_seq(backward_out, lengths)
        else:
            # inference 모드에서는 backward 사용 안함
            bsz, seq_len, _ = forward_out.size()
            backward_out = torch.zeros(bsz, seq_len, self.hidden_dim, device=forward_out.device)

        # forward_out과 backward_out concat
        combined_out = torch.cat([forward_out, backward_out], dim=2)
        rnn_output_embeddings = F.normalize(self.fc(combined_out), p=2, dim=2)

        if self.mode == 'train':
            # train 모드에서 loss 계산
            assert mask is not None, "train 모드에서는 mask 필요"
            total_loss = self.compute_total_loss(
                rnn_output_embeddings=rnn_output_embeddings,
                image_embeddings_normalized=image_embeddings_normalized,
                text_embeddings_normalized=text_embeddings_normalized,
                mask=mask
            )
            return rnn_output_embeddings, total_loss
        else:
            # inference 모드
            return rnn_output_embeddings

    def compute_total_loss(self, rnn_output_embeddings, image_embeddings_normalized, text_embeddings_normalized, mask):
        # Contrastive Loss 계산 및 정규화
        contrastive_loss = self.compute_contrastive_loss(
            image_embeddings_normalized, text_embeddings_normalized, mask
        )

        # Forward/Backward Loss 계산 및 정규화
        forward_loss, backward_loss = self.compute_rnn_loss(rnn_output_embeddings, mask)
        # num_rnn_samples = (mask[:, :-1].bool() & mask[:, 1:].bool()).sum().item()  # Forward/Backward Loss에서 유효 샘플 수
        # if num_rnn_samples > 0:
        #     forward_loss /= num_rnn_samples
        #     backward_loss /= num_rnn_samples

        # 총 손실 계산
        total_loss = self.contrastive_loss_factor * contrastive_loss + forward_loss + backward_loss
        #print(contrastive_loss, forward_loss, backward_loss, num_rnn_samples)
        print(rnn_output_embeddings.shape)
        return total_loss


    def compute_contrastive_loss(self, embeddings1, embeddings2, mask):
        bsz, seq_len, emb_dim = embeddings1.size()
        mask_flat = mask.view(-1).bool()

        emb1_flat = embeddings1.view(-1, emb_dim)[mask_flat]
        emb2_flat = embeddings2.view(-1, emb_dim)[mask_flat]

        scores = torch.matmul(emb1_flat, emb2_flat.T)
        diag = scores.diag().unsqueeze(1)

        cost_s = F.relu(self.emb_margin - diag + scores)
        cost_im = F.relu(self.emb_margin - diag.T + scores)

        identity_mask = torch.eye(cost_s.size(0), device=cost_s.device).bool()
        cost_s = cost_s.masked_fill(identity_mask, 0)
        cost_im = cost_im.masked_fill(identity_mask, 0)

        loss = (cost_s.sum() + cost_im.sum()) / (cost_s.size(0) ** 2)
        return loss

    def compute_rnn_loss(self, rnn_output_embeddings, mask):
        # Forward loss
        forward_input = rnn_output_embeddings[:, :-1, :]
        forward_target = rnn_output_embeddings[:, 1:, :]
        forward_mask = mask[:, :-1].bool() & mask[:, 1:].bool()  # boolean 변환
        forward_loss = self._compute_single_rnn_loss(forward_input, forward_target, forward_mask)

        # Backward loss
        backward_input = rnn_output_embeddings[:, 1:, :]
        backward_target = rnn_output_embeddings[:, :-1, :]
        backward_loss = self._compute_single_rnn_loss(backward_input, backward_target, forward_mask)

        return forward_loss, backward_loss

    def _compute_single_rnn_loss(self, input_embeddings, target_embeddings, mask):
        input_valid = input_embeddings[mask.bool()]
        target_valid = target_embeddings[mask.bool()]

        if input_valid.size(0) == 0:
            return torch.tensor(0.0, device=input_embeddings.device)

        scores = torch.matmul(input_valid, target_valid.T)
        targets = torch.arange(scores.size(0), device=scores.device)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(scores, targets)

    def _run_lstm_forward(self, image_embeddings, text_embeddings, lengths):
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=2)
        packed_inputs = pack_padded_sequence(
            combined_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.lstm(packed_inputs)
        lstm_out, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=combined_embeddings.size(1))
        return lstm_out

    def _reverse_seq(self, x, lengths):
        bsz, seq_len, dim = x.size()
        x_reversed = torch.zeros_like(x)
        for i in range(bsz):
            l = lengths[i].item()
            x_reversed[i, :l] = x[i, :l].flip(dims=[0])
        return x_reversed
