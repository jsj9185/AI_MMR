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
        for image_filename in image_filenames[1:]:  # Exclude 0.jpg
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

        # Encode texts and images
        text_input = clip.tokenize(text_list).to(self.device)
        texts = self.model.encode_text(text_input)
        image_tensor = self.preprocess_images(images).to(self.device)
        images = self.model.encode_image(image_tensor)

        if self.mode == 'test' and self.question is not None:
            filtered_question = self.question[self.question['set_id'] == set_id]
            question_data = {}
            for _, row in filtered_question.iterrows():
                question_data = {
                    'question_ids': row['question'],
                    'answer': row['answers'],
                    'blank_position': row['blank_position']
                }
            return {
                'texts': texts.float(),
                'images': images.float(),
                'set_id': set_id,
                'question': question_data,
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