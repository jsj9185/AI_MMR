
import torch
import torch.nn.functional as F

def pick_questions(texts, images, blank_idx):
    # blank 이전, blank 이후 부분을 이어붙여 blank를 제거한 시퀀스 생성
    texts_before_blank = texts[:, :blank_idx, :]
    texts_after_blank = texts[:, blank_idx+1:, :]
    images_before_blank = images[:, :blank_idx, :]
    images_after_blank = images[:, blank_idx+1:, :]
    q_texts = torch.cat((texts_before_blank, texts_after_blank), dim=1)
    q_images = torch.cat((images_before_blank, images_after_blank), dim=1)
    return q_texts, q_images

def get_question_embedding(rnn_output_embeddings, blank_idx, direction=2):
    """
    blank_idx: 0-based index of the blank in the original sequence.
    direction:
      - 1: forward only
      - -1: backward only
      - 2: bidirectional (average)
    rnn_output_embeddings: [batch, seq_len-1, embedding_size]
      blank 제거 후의 시퀀스 임베딩 (inference 결과)

    반환값: question_embedding (shape: [batch, embedding_size])
    """
    bsz, seq_len, emb_dim = rnn_output_embeddings.size()

    # blank가 첫번째 아이템이었다면 blank 제거 후 시퀀스의 첫 아이템이 backward 정보의 핵심
    # blank가 마지막 아이템이었다면 blank 제거 후 시퀀스의 마지막 아이템이 forward 정보의 핵심
    # 중간이라면 forward는 blank 이전 마지막 토큰, backward는 blank 이후 첫 토큰 사용
    # blank 제거 후의 시퀀스에서 blank_idx 기준:
    #   blank 이전 아이템 개수: blank_idx
    #   blank 이후 아이템 개수: (원래 길이 - blank_idx - 1)

    # forward embedding: blank 이전 마지막 토큰 (blank_idx-1)
    # backward embedding: blank 이후 첫 번째 토큰 (blank_idx) 
    # 단, blank가 첫 아이템이면 forward 없음, blank가 마지막 아이템이면 backward 없음

    # blank가 sequence 시작일 경우: forward 없음 -> backward only
    # blank가 sequence 끝일 경우: backward 없음 -> forward only
    # blank가 중간일 경우: 둘 다 존재

    forward_embedding = None
    backward_embedding = None

    # blank 전 아이템이 1개 이상 있으면 forward 임베딩 추출 가능
    if blank_idx > 0:
        forward_embedding = rnn_output_embeddings[:, blank_idx-1, :]  # blank 이전 마지막 아이템
    # blank 이후 아이템이 1개 이상 있으면 backward 임베딩 추출 가능
    # blank_idx는 original, 제거 후에도 동일 인덱스로 사용(앞에서 blank 제거 후 시퀀스 구성 시 동일 index)
    # 사실상 blank_idx 이후 아이템들은 그대로 뒤에 오므로 첫 아이템은 blank_idx 위치
    if blank_idx < seq_len:
        backward_embedding = rnn_output_embeddings[:, blank_idx, :]  # blank 이후 첫 아이템

    if direction == 1:
        # forward only
        if forward_embedding is not None:
            return forward_embedding
        else:
            # forward 없음 -> fallback backward
            return backward_embedding
    elif direction == -1:
        # backward only
        if backward_embedding is not None:
            return backward_embedding
        else:
            # backward 없음 -> fallback forward
            return forward_embedding
    else:
        # bidirectional (2)
        # 둘 다 존재하면 평균, 하나만 존재하면 그거 사용
        if forward_embedding is not None and backward_embedding is not None:
            return (forward_embedding + backward_embedding) / 2.0
        elif forward_embedding is not None:
            return forward_embedding
        else:
            return backward_embedding

def compute_scores(question_embedding, answer_embeddings):
    """
    question_embedding: [batch, embedding_size]
    answer_embeddings: [num_answers, embedding_size]
    dot product 후 softmax
    """
    # (batch, emb) dot (emb, num_answers) -> (batch, num_answers)
    scores = torch.matmul(question_embedding, answer_embeddings.T)
    probs = F.softmax(scores, dim=1)  # 확률화
    return probs

def fill_in_blank_inference(model, batch, device, direction=2):
    """
    model: PolyvoreLSTMModel(mode='inference') 인스턴스
    batch: {'texts': [batch, seq_len, 512], 'images': [...], 'question': {'blank_position', 'answer'}}
    direction: 1 (forward), -1 (backward), 2 (bidirectional)
    """
    texts = batch['texts'].to(device)
    images = batch['images'].to(device)
    blank_idx = batch['question']['blank_position'].item() - 1  # 0-based index
    answer_sheet = batch['question']['answer']  # list of 4 candidates: [(set_id_idx, ), ...]
    lengths = torch.tensor([texts.size(1)-1], dtype=torch.long).to(device)  # blank 제거 후 길이

    # blank 제거한 질문 시퀀스
    q_texts, q_images = pick_questions(texts, images, blank_idx)

    # 모델 추론
    rnn_output_embeddings = model(q_images, q_texts, lengths)  # inference 모드 -> rnn_output_embeddings만 반환

    # 질문 임베딩 추출
    question_embedding = get_question_embedding(rnn_output_embeddings, blank_idx, direction=direction)

    # 답변 후보 임베딩 추출 (가정: answer_embeddings를 미리 얻을 수 있음)
    # answer_sheet: 예) [('100_3',), ('100_5',), ('200_1',), ('300_2',)]
    # 각 답변에 대해 image+text를 하나의 아이템으로 처리 -> (512+512)=1024차원 임베딩 가정
    # 여기서는 단순히 answer_embeddings를 미리 준비했다고 가정
    # answer_embeddings: [4, embedding_size]

    answer_embeddings = []
    for ans in answer_sheet:
        # ans: ('setid_idx',)
        value = ans[0]
        set_id, idx = value.split('_')
        set_id = int(set_id)
        idx = int(idx)
        # test_dataset를 통해 답변 아이템 임베딩 가져오기 가정 (image+text concat)
        # answer_item_embedding = ... # shape: [1024]
        # 여기서는 가짜 임베딩:
        answer_item_embedding = torch.randn(1, model.embedding_size).to(device)
        answer_embeddings.append(answer_item_embedding)

    answer_embeddings = torch.cat(answer_embeddings, dim=0)  # [4, embedding_size]

    probs = compute_scores(question_embedding, answer_embeddings)  # [1,4]
    predicted_answer = torch.argmax(probs, dim=1)  # 정답 인덱스
    return predicted_answer.item()

### 사용 예시
# model = PolyvoreLSTMModel(mode='inference', ...)
# model.to(device)
# model.eval()
# batch = next(iter(test_loader)) # 예시
# answer_idx = fill_in_blank_inference(model, batch, device, direction=2)
# print("Predicted answer index:", answer_idx)
