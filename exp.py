import pandas as pd
from dataloader.sampling import DataSampler
#from dataloader.multimodal_data import MultiModalData
from dataloader.multilstm_data import MultiModalData
import os

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'data')
meta_dir = os.path.join(data_dir, 'meta')
image_dir = os.path.join(data_dir, 'images')
sampler = DataSampler(data_path = meta_dir, k=100, test_sampling_ratio=1)
concat_df, question_data = sampler.sample_data()

#train_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='train')
#valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')

print(test_dataset.count1, test_dataset.count2)
print(concat_df['type'].value_counts())