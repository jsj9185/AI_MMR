import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from dataloader.sampling import DataSampler
from dataloader.multilstm_data import MultiModalData
from model.multilstm import PolyvoreLSTMModel
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

def pick_questions_v3(texts, images, blank_idx):
    """
    Blank Position 기준으로 Forward와 Backward Context를 분리.
    """
    forward_texts = texts[:, :blank_idx+1, :]     # Blank 이전 텍스트
    backward_texts = texts[:, blank_idx:, :]  # Blank 이후 텍스트
    forward_images = images[:, :blank_idx+1, :]   # Blank 이전 이미지
    backward_images = images[:, blank_idx:, :]  # Blank 이후 이미지

    return forward_texts, forward_images, backward_texts, backward_images

def compute_probs(context_embedding, candidate_embeddings): # context embedding 하고matmul할거냐 or 1대1로 matmul을 할거냐
    """
    주어진 Context Embedding과 Candidate Embedding으로 Softmax 확률 계산.
    """
    candidate_embeddings = candidate_embeddings.squeeze(0)
    #print(context_embedding)
    #print(context_embedding.shape, candidate_embeddings.shape)
    #scores = F.cosine_similarity(context_embedding, candidate_embeddings)
    scores = torch.matmul(context_embedding, candidate_embeddings.T)  # Dot Product
    #print(scores)
    probs = F.softmax(scores, dim=1)  # Softmax 계산
    return probs

def inference(model, test_dataloader, checkpoint_path, device):
    """
    Inference 함수: 모델을 이용하여 Fill-in-the-Blank 태스크를 수행.
    """
    # 모델 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("############ Inference Start #############\n")
    
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # Data 준비
            texts = batch['texts'].to(device)   # [batch, seq_len, embedding_dim]
            images = batch['images'].to(device)
            question = batch['questions'][0]
            #print(question)
            blank_idx = question['blank_position'] - 1  # Blank Position (1-based -> 0-based)
            answer_texts = batch['answer_texts'].to(device)  # [num_candidates, embedding_dim]
            answer_images = batch['answer_images'].to(device)  # [num_candidates, embedding_dim]

            if answer_texts.shape != torch.Size([1, 4, 512]):
                continue
            #print(blank_idx)
            # Answer 후보 임베딩 생성
            answer_embeddings = model.combine_embeddings(answer_texts, answer_images)  # [1, 4, 512]

            # Forward/Backward Context를 위한 시퀀스 분리
            forward_texts, forward_images, backward_texts, backward_images = pick_questions_v3(texts, images, blank_idx)

            # Forward Context 사용 여부
            use_forward = (blank_idx > 0)  # Blank가 첫 번째 아이템이 아니면 사용 가능
            # Backward Context 사용 여부
            use_backward = (blank_idx < texts.size(1) - 1)  # Blank가 마지막 아이템이 아니면 사용 가능

            # Forward Context 계산
            if use_forward:
                forward_lengths = torch.tensor([forward_texts.size(1)], dtype=torch.long, device=device)
                forward_context, _ = model(forward_images, forward_texts, forward_lengths)
                #forward_context = forward_out[:, -1, :]  # Forward Context: 마지막 유효 토큰의 임베딩
            else:
                forward_context = None

            # Backward Context 계산
            if use_backward:
                backward_lengths = torch.tensor([backward_texts.size(1)], dtype=torch.long, device=device)
                #rev_backward_texts = torch.flip(backward_texts, dims=[1])
                #rev_backward_images = torch.flip(backward_images, dims=[1])
                rev_backward_images = model._reverse_seq(backward_images, backward_lengths)
                rev_backward_texts = model._reverse_seq(backward_texts, backward_lengths)
                _, backward_context = model(rev_backward_images, rev_backward_texts, backward_lengths)
                #backward_context = backward_out[:, -1, :]  # Backward Context: 마지막 유효 토큰의 임베딩
            else:
                backward_context = None

            # Forward와 Backward Context 합성
            if forward_context is not None and backward_context is not None:
                total_length = (forward_lengths.item() + backward_lengths.item())
                forward_weight = forward_lengths.item() / total_length
                backward_weight = backward_lengths.item() / total_length
                forward_probs = compute_probs(forward_context, answer_embeddings)
                backward_probs = compute_probs(backward_context, answer_embeddings)
                probs = forward_weight * forward_probs + backward_weight * backward_probs  # 합산
            elif forward_context is not None:
                probs = compute_probs(forward_context, answer_embeddings)  # Forward만 사용
            elif backward_context is not None:
                probs = compute_probs(backward_context, answer_embeddings)  # Backward만 사용
            else:
                raise ValueError("Both forward and backward contexts are None.")

            # 후보 정답에 대한 확률 계산
            #print(probs)
            pred_idx = torch.argmax(probs, dim=1).item()  # 가장 높은 확률의 정답 후보 선택
            predicted_answer = question['answer'][pred_idx]  # 선택된 정답

            # 정답 검증
            if predicted_answer == question['answer'][0]:  # 첫 번째가 Ground Truth로 가정
                print(f"Batch {batch_idx+1}: Predicted Answer = {predicted_answer}, Correct Answer = {question['answer'][0]}")
                correct += 1
            total += 1

            #print(f"Batch {batch_idx+1}: Predicted Answer = {predicted_answer}, Correct Answer = {question['answer'][0]}")

    # 최종 정확도 출력
    accuracy = correct / total * 100
    print(f"########### Final Accuracy: {accuracy:.2f}% ({correct}/{total}) ###########")
    return accuracy

from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable sequence lengths.
    """
    # 각 요소별 데이터 수집
    texts = [item['texts'] for item in batch]  # 텍스트 임베딩
    images = [item['images'] for item in batch]  # 이미지 임베딩
    lengths = torch.tensor([len(t) for t in texts])  # 각 샘플의 시퀀스 길이
    set_ids = [item['set_id'] for item in batch]  # set_id는 리스트로 유지
    questions = [item['question'] for item in batch]  # 질문 정보
    answer_texts = [item['answer_texts'] for item in batch]  # 정답 후보 텍스트
    answer_images = [item['answer_images'] for item in batch]  # 정답 후보 이미지

    # 시퀀스 데이터 패딩
    texts_padded = pad_sequence(texts, batch_first=True)  # [batch_size, max_seq_len, embedding_dim]
    images_padded = pad_sequence(images, batch_first=True)  # [batch_size, max_seq_len, embedding_dim]

    # 정답 후보를 텍스트와 이미지로 각각 패딩
    max_num_candidates = max(a.size(0) for a in answer_texts)  # 가장 많은 후보 수 찾기
    padded_answer_texts = torch.zeros(len(batch), max_num_candidates, texts[0].size(1))  # [batch_size, max_candidates, embedding_dim]
    padded_answer_images = torch.zeros(len(batch), max_num_candidates, images[0].size(1))  # [batch_size, max_candidates, embedding_dim]

    for i in range(len(batch)):
        padded_answer_texts[i, :answer_texts[i].size(0), :] = answer_texts[i]
        padded_answer_images[i, :answer_images[i].size(0), :] = answer_images[i]

    return {
        'texts': texts_padded,
        'images': images_padded,
        'lengths': lengths,
        'set_ids': set_ids,
        'questions': questions,
        'answer_texts': padded_answer_texts,
        'answer_images': padded_answer_images,
    }

class Config:
    embedding_size = 512
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.005
    num_epochs = 20
    batch_size = 1
    #contrastive_loss_factor = 1.0
    category_k = 150

config = Config()
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "checkpoints/checkpoint_12092321/best_checkpoint.pth"  # 저장된 체크포인트 경로
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    meta_dir = os.path.join(data_dir, 'meta')
    image_dir = os.path.join(data_dir, 'images')
    sampler = DataSampler(data_path = meta_dir, k=config.category_k, test_sampling_ratio=1.0)
    concat_df, question_data = sampler.sample_data()
    test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')
    print(len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # test_dataset은 이미 로드된 상태 가정

    # 모델 초기화
    model = PolyvoreLSTMModel(
        embedding_size=config.embedding_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        #contrastive_loss_factor=config.contrastive_loss_factor,
        mode='inference'
    ).to(device)
    print("############## Model Loaded ################")
    
    # 추론 실행
    inference(model, test_dataloader, checkpoint_path, device)


'''import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from dataloader.sampling import DataSampler
from dataloader.multilstm_data import MultiModalData
from model.multilstm import PolyvoreLSTMModel
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

def pick_questions_v3(texts, images, blank_idx):
    """
    Blank Position 기준으로 Forward와 Backward Context를 분리.
    """
    forward_texts = texts[:, :blank_idx, :]     # Blank 이전 텍스트
    backward_texts = texts[:, blank_idx+1:, :]  # Blank 이후 텍스트
    forward_images = images[:, :blank_idx, :]   # Blank 이전 이미지
    backward_images = images[:, blank_idx+1:, :]# Blank 이후 이미지

    return forward_texts, forward_images, backward_texts, backward_images

def compute_probs(context_embedding, candidate_embeddings):
    """
    단순 Cosine similarity 기반 후보 선택.
    context_embedding: [B, H] - 여기서 B=1이라고 가정.
    candidate_embeddings: [1, num_candidates, H]
    """
    candidate_embeddings = candidate_embeddings.squeeze(0)  # [num_candidates, H]
    # Cosine similarity 계산
    #print(f"context_embedding type: {type(context_embedding)}")
    #print(f"candidate_embeddings type: {type(candidate_embeddings)}")
    scores = F.cosine_similarity(context_embedding, candidate_embeddings)
    # Softmax로 확률화
    probs = F.softmax(scores, dim=0)
    return probs

def inference(model, test_dataloader, checkpoint_path, device):
    """
    Inference 함수: 모델을 이용하여 Fill-in-the-Blank 태스크를 수행.
    새로운 로직:
    1. Forward Context (1~t-1)로 t 위치 임베딩 예측 -> k_hat
    2. Backward Context (t+1~N) 역순으로 입력 -> t 위치 임베딩 예측 -> reversed_1_hat
    3. 두 임베딩 가중합(또는 평균) -> 최종 context_embedding
    4. 후보들 중 가장 유사한 정답 선택
    """
    # 모델 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("############ Inference Start #############\n")
    
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            texts = batch['texts'].to(device)   # [batch=1, seq_len, embed_dim]
            images = batch['images'].to(device)
            question = batch['questions'][0]
            blank_idx = question['blank_position'] - 1  # Blank Position (1-based -> 0-based)
            answer_texts = batch['answer_texts'].to(device)  # [1, num_candidates, embed_dim]
            answer_images = batch['answer_images'].to(device) # [1, num_candidates, embed_dim]
            #print(question)
            #print(texts.shape, images.shape)
            # 후보 임베딩 계산
            answer_embeddings = model.combine_embeddings(answer_texts, answer_images)  # [1, 4, H]

            # Forward/Backward Context 나누기
            forward_texts, forward_images, backward_texts, backward_images = pick_questions_v3(texts, images, blank_idx)

            use_forward = (blank_idx > 0)
            use_backward = (blank_idx < texts.size(1) - 1)

            # Forward Context로 예측 (k_hat)
            if use_forward:
                forward_lengths = torch.tensor([forward_texts.size(1)], dtype=torch.long, device=device)
                # forward 호출: (forward_images, forward_texts, forward_lengths)
                # 새 로직 가정: model(...) 호출 시 forward context -> next token 임베딩 (k_hat) 반환
                k_hat, _ = model(forward_images, forward_texts, forward_lengths)  # [B, H], B=1
            else:
                k_hat = None

            # Backward Context로 예측 (reversed_1_hat)
            if use_backward:
                backward_lengths = torch.tensor([backward_texts.size(1)], dtype=torch.long, device=device)
                rev_backward_images = model._reverse_seq(backward_images, backward_lengths)
                rev_backward_texts = model._reverse_seq(backward_texts, backward_lengths)
                _, reversed_1_hat = model(rev_backward_images, rev_backward_texts, backward_lengths)  # [B, H]
            else:
                reversed_1_hat = None

            # Forward와 Backward Context 결합
            if k_hat is not None and reversed_1_hat is not None:
                total_length = forward_lengths.item() + backward_lengths.item()
                forward_weight = forward_lengths.item() / total_length
                backward_weight = backward_lengths.item() / total_length
                # 가중 평균
                forward_probs = compute_probs(k_hat, answer_embeddings)
                backward_probs = compute_probs(reversed_1_hat, answer_embeddings)
                probs = forward_weight * forward_probs + backward_weight * backward_probs
            elif k_hat is not None:
                probs = compute_probs(k_hat, answer_embeddings)
            elif reversed_1_hat is not None:
                probs = compute_probs(reversed_1_hat, answer_embeddings)
            else:
                raise ValueError("No forward or backward context available.")
            pred_idx = torch.argmax(probs, dim=0).item()
            predicted_answer = question['answer'][pred_idx]

            # 정답 검증(첫 번째 answer가 정답이라고 가정)
            if predicted_answer == question['answer'][0]:
                correct += 1
            total += 1
            print(f"Batch {batch_idx+1}: Predicted = {predicted_answer}, GT = {question['answer'][0]}")

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"########### Final Accuracy: {accuracy:.2f}% ({correct}/{total}) ###########")
    return accuracy


class Config:
    embedding_size = 512
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.01
    num_epochs = 20
    batch_size = 1
    category_k = 150

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "checkpoints/checkpoint_12092202/best_checkpoint.pth"
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    meta_dir = os.path.join(data_dir, 'meta')
    image_dir = os.path.join(data_dir, 'images')

    sampler = DataSampler(data_path=meta_dir, k=Config.category_k, test_sampling_ratio=1.0)
    concat_df, question_data = sampler.sample_data()
    test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question=question_data, mode='test')

    def collate_fn(batch):
        texts = [item['texts'] for item in batch]
        images = [item['images'] for item in batch]
        lengths = torch.tensor([len(t) for t in texts])
        set_ids = [item['set_id'] for item in batch]
        questions = [item['question'] for item in batch]
        answer_texts = [item['answer_texts'] for item in batch]
        answer_images = [item['answer_images'] for item in batch]

        texts_padded = pad_sequence(texts, batch_first=True)
        images_padded = pad_sequence(images, batch_first=True)
        max_num_candidates = max(a.size(0) for a in answer_texts)
        padded_answer_texts = torch.zeros(len(batch), max_num_candidates, texts[0].size(1))
        padded_answer_images = torch.zeros(len(batch), max_num_candidates, images[0].size(1))

        for i in range(len(batch)):
            padded_answer_texts[i, :answer_texts[i].size(0), :] = answer_texts[i]
            padded_answer_images[i, :answer_images[i].size(0), :] = answer_images[i]

        return {
            'texts': texts_padded,
            'images': images_padded,
            'lengths': lengths,
            'set_ids': set_ids,
            'questions': questions,
            'answer_texts': padded_answer_texts,
            'answer_images': padded_answer_images,
        }

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = PolyvoreLSTMModel(
        embedding_size=Config.embedding_size,
        hidden_dim=Config.hidden_dim,
        num_layers=Config.num_layers,
        dropout=Config.dropout,
        mode='inference'
    ).to(device)

    inference(model, test_dataloader, checkpoint_path, device)
'''