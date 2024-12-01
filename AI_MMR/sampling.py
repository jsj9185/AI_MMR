import pandas as pd
import os

class DataSampler:
    """
    Arguements: ~
    """
    def __init__(self, data_path, k=100, train_sampling_ratio=1.0, test_sampling_ratio = 0.33):
        self.data_path = data_path
        self.k = k
        self.train_sampling_ratio = train_sampling_ratio
        self.test_sampling_ratio = test_sampling_ratio
        self.train = pd.read_json(os.path.join(self.data_path, 'train_no_dup.json'))
        self.valid = pd.read_json(os.path.join(self.data_path, 'valid_no_dup.json'))
        self.test = pd.read_json(os.path.join(self.data_path, 'test_no_dup.json'))
        self.question = pd.read_json(os.path.join(self.data_path, 'fill_in_blank_test.json'))
        self.category_df = self._load_category_data(os.path.join(self.data_path, 'category_id.txt'))
        self.top_categories = None

    def _load_category_data(self, category_path):
        category = []
        with open(category_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                category.append([parts[0], " ".join(parts[1:])])
        return pd.DataFrame(category, columns=['ID', 'Category'])

    def _get_top_categories(self, df):
        category_counts = {}
        for _, row in df.iterrows():
            for item in row['items']:
                category_id = item['categoryid']
                if category_id in category_counts:
                    category_counts[category_id] += 1
                else:
                    category_counts[category_id] = 1

        sorted_category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
        top_k_categories = list(sorted_category_counts.keys())[:self.k]
        return top_k_categories

    def _filter_data(self, df):
        top_categories = self.top_categories
        df = df[['name', 'items', 'desc', 'set_id']]
        def keep_row(row):
            for item in row['items']:
                if item['categoryid'] not in top_categories:
                    return False
            return True

        return df[df.apply(keep_row, axis=1)]

    def get_matching_questions_data(self, blank_df):
        test_set_ids = set(self.test['set_id'].values)
        blank_df['set_id'] = blank_df['question'].apply(
            lambda questions: int(questions[0].split('_')[0]) if len(questions) > 0 and '_' in questions[0] else None
        )
        matching_blanks = blank_df[blank_df['set_id'].isin(test_set_ids)]
        return matching_blanks

    def sample_data(self):
        self.top_categories = self._get_top_categories(self.train)
        self.train = self._filter_data(self.train)
        self.valid = self._filter_data(self.valid)
        self.test = self._filter_data(self.test)

        self.train = self.train.sample(frac=self.train_sampling_ratio, random_state=42)
        self.test = self.test.sample(frac=self.test_sampling_ratio, random_state=42)
        self.train['type'] = 'train'
        self.valid['type'] = 'valid'
        self.test['type'] = 'test'

        self.question = self.get_matching_questions_data(self.question)

        concat_df = pd.concat([self.train, self.valid, self.test], ignore_index=True)

        return concat_df, self.question
    
'''
from importlib import reload
import sampling  
reload(sampling) 
from sampling import DataSampler 
import os

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'data')
meta_dir = os.path.join(data_dir, 'meta')
image_dir = os.path.join(data_dir, 'images')
sampler = DataSampler(data_path = meta_dir, sampling_ratio=0.5)
concat_df, question_data = sampler.sample_data()
'''