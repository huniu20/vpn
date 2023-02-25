import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from param import Param
from data_process import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Vpn
from torch.nn import CrossEntropyLoss

def main():
    p = Param()
    x = load_dataset(p.test_file, p)
    train_data = build_features(x, p)
    # print(feature_map)
    train_loader = DataLoader(dataset=train_data,batch_size=p.batch_size,shuffle=True,collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Vpn(p)
    model.to(device)
    loss_fn = CrossEntropyLoss()
    for batch_data in tqdm(train_loader):
        input_ids, label_ids, attention_mask = map(lambda x: x.to(device), batch_data)
        output = model(input_ids, label_ids, attention_mask)
        print(batch_data)
        break

if __name__ == "__main__":
    p = Param()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x = load_dataset(p.test_file, p)
    # train_data = build_features(x, p)
    # print(feature_map)
    # train_loader = DataLoader(dataset=train_data,batch_size=p.batch_size,shuffle=True,collate_fn=collate_fn)
    # for batch_data in tqdm(train_loader):
    #     print((batch_data))
    #     break
    input_ids = torch.LongTensor([ 101, 2769, 1373, 6627, 3173, 3696, 2769,  738, 6206, 1391, 3165, 1157,
         1157, 1962, 1450, 5489, 1259, 5291, 8043, 6821, 6573, 4994, 4197, 3221,
         5489, 1259, 5291, 1922, 5846,  749, 1416,  840, 2157, 6963,  679, 5650,
         2533, 1391, 1568, 1599, 3614, 1391, 8024, 2218, 1068, 3800, 1391, 6573,
         6963, 1762, 6821, 7027, 1256, 3022, 1416,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,  102])
    attention_mask = torch.LongTensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.])
    label_ids = torch.LongTensor([-100,    0,    0,    1,    2,    2,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    2,
            0,    0,    0,    0,    0,    0,    0, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100])
    
    
    model = Vpn(p)
    # print(model)
    for name, param in model.named_parameters():
        if "CRF" or "crf" in name:
            print(name,type(param), param.size())
