import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import clip

class MultiModalData(Dataset):
    def __init__(self, data_df, category_df, image_dir, question=None, mode='train'):
        self.mode = mode
        self.image_dir = image_dir
        self.question = question
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.preprocess = preprocess
        self.model = model
        self.embedding_size = 512
        self.count1=0
        self.count2=0

        # Category Dictionary
        self.category_dict = category_df.set_index('ID')['Category'].to_dict()

        # Filter valid data rows with non-empty image sequences
        self.data_df = self.filter_valid_data(data_df[data_df['type'] == mode].reset_index(drop=True))

        # Map set_id to indices for question processing
        self.set_id_search_dict = {row['set_id']: idx for idx, row in self.data_df.iterrows()}

    def filter_valid_data(self, df):
        """
        Filters rows with valid image sequences.
        """
        valid_rows = []
        for idx, row in df.iterrows():
            image_folder = os.path.join(self.image_dir, str(row['set_id']))
            if os.path.isdir(image_folder):
                image_filenames = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
                if len(image_filenames) > 1:  # At least one valid image (excluding 0.jpg)
                    valid_rows.append(idx)
            else:
                print(f"Invalid or missing image folder: {image_folder}")
        return df.iloc[valid_rows].reset_index(drop=True)

    def preprocess_images(self, image_list):
        """
        Preprocesses images using CLIP's preprocess function.
        """
        processed_images = [self.preprocess(image).unsqueeze(0) for image in image_list]
        return torch.cat(processed_images, dim=0)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data_row = self.data_df.iloc[idx]
        set_id = data_row['set_id']

        # Generate text list
        text_list = []
        items = data_row['items']
        for item in items:
            category_id = str(item['categoryid'])
            category_name = self.category_dict.get(category_id, 'Unknown Category')
            item_name = item['name']
            text_list.append(f"{category_name}: {item_name}.")

        # Load images
        images = []
        image_folder = os.path.join(self.image_dir, str(set_id))
        image_filenames = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
        for image_filename in image_filenames[1:]:  # 1.jpg부터 아이템으로 간주
            image_path = os.path.join(image_folder, image_filename)
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Error loading image: {image_path}, Error: {e}")

        # Match sequence length between texts and images
        max_length = min(len(text_list), len(images))
        text_list = text_list[:max_length]
        images = images[:max_length]

        # Encode texts and images for main items
        text_input = clip.tokenize(text_list).to(self.device)
        texts = self.model.encode_text(text_input)
        image_tensor = self.preprocess_images(images).to(self.device)
        images = self.model.encode_image(image_tensor)

        # Test 모드에서 questions 처리 및 answer 임베딩 생성
        if self.mode == 'test' and self.question is not None:
            filtered_question = self.question[self.question['set_id'] == set_id]
            question_data = {}
            answer_texts = []
            answer_images = []
            
            for _, q_row in filtered_question.iterrows():
                question_data = {
                    'question_ids': q_row['question'],
                    'answer': q_row['answers'],
                    'blank_position': q_row['blank_position'],
                }

                # answer 예: ['189533874_6', '215389587_2', ...]
                # 각 answer_id 파싱
                for ans_id in question_data['answer']:
                    ans_set_id_str, ans_item_idx_str = ans_id.split('_')
                    ans_set_id = int(ans_set_id_str)
                    ans_item_idx = int(ans_item_idx_str)

                    # 해당 set_id에 해당하는 row 찾기
                    ans_set_idx = self.set_id_search_dict.get(ans_set_id, None)
                    if ans_set_idx is None:
                        #print(f"Warning: set_id {ans_set_id} not found, skipping candidate {ans_id}")
                        self.count1+=1
                        continue

                    answer_data_row = self.data_df.iloc[ans_set_idx]
                    ans_items = answer_data_row['items']

                    if ans_item_idx > len(ans_items):
                        self.count2+=1
                        #print(f"Warning: item idx {ans_item_idx} out of range for set_id {ans_set_id}, skipping {ans_id}")
                        continue

                    ans_item = ans_items[ans_item_idx - 1]
                    category_id = str(ans_item['categoryid'])
                    category_name = self.category_dict.get(category_id, 'Unknown Category')
                    item_name = ans_item['name']
                    ans_text = f"{category_name}: {item_name}."
                    answer_texts.append(ans_text)

                    # 이미지 로드
                    ans_image_path = os.path.join(self.image_dir, str(ans_set_id), f"{ans_item_idx}.jpg")
                    try:
                        ans_image = Image.open(ans_image_path).convert('RGB')
                        ans_image_tensor = self.preprocess(ans_image).unsqueeze(0).to(self.device)
                        ans_image_emb = self.model.encode_image(ans_image_tensor)
                        answer_images.append(ans_image_emb)
                    except Exception as e:
                        print(f"Error loading answer image: {ans_image_path}, Error: {e}")

            # answer_texts와 answer_images 임베딩
            if answer_texts:
                answer_text_input = clip.tokenize(answer_texts).to(self.device)
                answer_text_emb = self.model.encode_text(answer_text_input)

            if answer_images:
                answer_images_emb = torch.cat(answer_images, dim=0)  # [num_answers, emb_dim]


            return {
                'texts': texts.float(),
                'images': images.float(),
                'set_id': set_id,
                'question': question_data,
                'answer_texts': answer_text_emb.float(),
                'answer_images': answer_images_emb.float()
            }

        return {
            'texts': texts.float(),
            'images': images.float(),
            'set_id': set_id,
        }



'''
from importlib import reload
import multimodal_data 
reload(multimodal_data)
from multimodal_data import MultiModalData

#train_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='train')
#valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')
'''


'''class TransformerModel():

    def __init__(self, config, mode):
        """
        Arguements: ~

        """
        self.config = config
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = self.config['embedding_dim']

        self.clip_model, self.clip_preprocess = clip.load(self.config['clip_model_name'], self.device)

    def load_data(self, image_dir, metadata_file, batch_size, shuffle=False, sampler=None):
        dataset = Dataset(image_dir, metadata_file, self.clip_preprocess)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=4)

        return data_loader

    def image_embeddings(self):

        pass

    def text_embeddings(self):

        pass'''