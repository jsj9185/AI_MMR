import pandas as pd
import torch
import clip
from PIL import Image
from importlib import reload
from dataloader.sampling import DataSampler  # 이미 import된 모듈
from importlib import reload
from dataloader.multimodal_data import MultiModalData
import os
from torch.utils.data import Dataset, DataLoader
from model.multifusion import Multifusion

path = 'data/meta/'
train = pd.read_json(path + 'train_no_dup.json')
valid = pd.read_json(path + 'valid_no_dup.json')
test = pd.read_json(path + 'test_no_dup.json')
blank = pd.read_json(path + 'fill_in_blank_test.json')


base_dir = ''
data_dir = os.path.join(base_dir, 'data')
meta_dir = os.path.join(data_dir, 'meta')
image_dir = os.path.join(data_dir, 'images')
sampler = DataSampler(data_path = meta_dir,  test_sampling_ratio=0.33)
concat_df, question_data = sampler.sample_data()



train_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='train')
valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')


device = "cuda" if torch.cuda.is_available() else "cpu"
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
net = Multifusion().to(device)

for batch_idx, batch in enumerate(dataloader):
  
    print("images: " , batch['images'].shape)
    print("texts:" , batch['texts'].shape)
    print("prices: ", batch['prices'].shape)
    print("likes: ", batch['likes'].shape)
    print("valid_idx: ", batch['valid_idx'].shape)
    print("gt_idx: ", batch['gt_idx'].shape)
    fused_output = net(batch['images'], batch['texts'], batch['prices'], batch['likes'])
    print("fused_output: ", fused_output.shape)
    break
