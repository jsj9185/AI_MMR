import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from dataloader.sampling import DataSampler
from dataloader.multilstm_data import MultiModalData
from model.multilstm import PolyvoreLSTMModel

def pick_questions_v3(texts, images, blank_idx):
    """
    blank_idx 기준으로 forward: x_1 ... x_{t-1}
                        backward: x_{t+1} ... x_N
    """
    forward_texts = texts[:, :blank_idx, :]      # x_1 ... x_{t-1}
    backward_texts = texts[:, blank_idx+1:, :]   # x_{t+1} ... x_N

    forward_images = images[:, :blank_idx, :]
    backward_images = images[:, blank_idx+1:, :]

    return forward_texts, forward_images, backward_texts, backward_images

def compute_probs(context_embedding, candidate_embeddings):
    """
    context_embedding: [batch, embedding_dim]
    candidate_embeddings: [num_candidates, embedding_dim]

    softmax 확률 계산:
    P(x_c|context) = exp(context·x_c) / Σ_x exp(context·x)
    """
    scores = torch.matmul(context_embedding, candidate_embeddings.T)  # [batch, num_candidates]
    probs = F.softmax(scores, dim=1)  # 확률로 변환
    return probs

def inference(model, test_dataloader, checkpoint_path, device, embedding_size=512):
    # 모델 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            texts = batch['texts'].to(device)   # [batch, seq_len, embedding_dim]
            images = batch['images'].to(device)
            question = batch['question'][0]  # 단일 question 처리 가정
            blank_pos = question['blank_position']  # blank position (1-based)
            blank_idx = blank_pos - 1  # 0-based index
            answer_texts = batch['answer_texts'].to(device)  # [num_candidates, embedding_dim]
            answer_images = batch['answer_images'].to(device)  # [num_candidates, embedding_dim]

            # 후보 임베딩 계산 (text+image 평균화)
            answer_embeddings = (answer_texts + answer_images) / 2.0  # [num_candidates, embedding_dim]

            # forward/backward 시퀀스 분리
            forward_texts, forward_images, backward_texts, backward_images = pick_questions_v3(texts, images, blank_idx)

            # forward_context 계산 여부
            use_forward = (blank_idx > 1)  # blank가 첫 아이템이 아니면 forward 사용 가능
            # backward_context 계산 여부
            use_backward = (blank_idx < texts.size(1))  # blank가 마지막 아이템이 아니면 backward 사용 가능

            # forward_context
            if use_forward:
                forward_context = model.forward_only(forward_texts, forward_images)  # [batch, emb_dim]
            else:
                forward_context = None

            # backward_context
            if use_backward:
                rev_backward_texts = torch.flip(backward_texts, dims=[1])
                rev_backward_images = torch.flip(backward_images, dims=[1])
                backward_context = model.forward_only(rev_backward_texts, rev_backward_images)  # [batch, emb_dim]
            else:
                backward_context = None

            # forward/backward 확률 계산
            if use_forward and use_backward:
                # 둘 다 사용
                forward_probs = compute_probs(forward_context, answer_embeddings)
                backward_probs = compute_probs(backward_context, answer_embeddings)
                total_probs = forward_probs + backward_probs
            elif use_forward:
                # forward만 사용
                total_probs = compute_probs(forward_context, answer_embeddings)
            elif use_backward:
                # backward만 사용
                total_probs = compute_probs(backward_context, answer_embeddings)
            else:
                # forward도 backward도 없으면
                print("No forward or backward context available.")
                total_probs = torch.ones(1, len(answer_embeddings), device=device) / len(answer_embeddings)

            # argmax로 최종 후보 선택
            pred_idx = torch.argmax(total_probs, dim=1).item()
            predicted_answer = question['answer'][pred_idx]

            # 정답 검증
            if predicted_answer == question['answer'][0]:
                correct += 1
            total += 1

            print(f"Batch {batch_idx+1}: Predicted Answer = {predicted_answer}, Correct: {question['answer'][0]}")

    accuracy = correct / total * 100
    print(f"########### Final Accuracy: {accuracy:.2f}% ({correct}/{total}) ###########")
    return accuracy

# 사용 예시
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "checkpoints/checkpoint_12062101.pth"  # 저장된 체크포인트 경로

    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    meta_dir = os.path.join(data_dir, 'meta')
    image_dir = os.path.join(data_dir, 'images')
    sampler = DataSampler(data_path = meta_dir, k=150, test_sampling_ratio=1)
    concat_df, question_data = sampler.sample_data()
    test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # test_dataset은 이미 로드된 상태 가정

    # 모델 초기화
    model = PolyvoreLSTMModel(mode='inference', embedding_size=512, hidden_dim=256).to(device)

    # 추론 실행
    inference(model, test_dataloader, checkpoint_path, device)
