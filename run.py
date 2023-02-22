import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from param import Param
from data_process import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Vpn

def main():
    p = Param()
    x = load_dataset(p.test_file, p)
    train_data = build_features(x, p)
    # print(feature_map)
    train_loader = DataLoader(dataset=train_data,batch_size=p.batch_size,shuffle=True,collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Vpn(p)
    model.to(device)
    for batch_data in tqdm(train_loader):
        input_ids, label_ids, attention_mask = map(lambda x: x.to(device), batch_data)
        output = model(input_ids, label_ids, attention_mask)
        print(batch_data)
        break

if __name__ == "__main__":
    p = Param()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Vpn(p)
    print(model)
