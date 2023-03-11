import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from param import Param
from data_process import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Vpn
from torch.nn import CrossEntropyLoss
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import time

def main():
    time_tuple = time.localtime(time.time())
    p = Param()
    x = load_dataset(p.train_file, p)
    build_features_new = partial(build_features, param=p)
    # lr = p.lr
    # crf_lr = p.crf_lr
    with ProcessPoolExecutor(p.train_kernel) as pool:
        train_data = list(pool.map(build_features_new,x))
    input_ids, label_ids, attention_mask = list(zip(*train_data))
    train_data = NERDataset(input_ids, label_ids, attention_mask)
    # train_data = build_features(x, p)
    # print(feature_map)
    train_loader = DataLoader(dataset=train_data,batch_size=p.batch_size,shuffle=True,collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Vpn(p)
    model.to(device)
    # loss_fn = CrossEntropyLoss()
    crf_group = {"params":[], "lr":1e-3}
    other_group = {"params":[], "lr":1e-5}
    for name, para in model.named_parameters():
        if "crf" in name:
            crf_group["params"].append(para)
        else:
            other_group["params"].append(para)
    optimizer = torch.optim.Adam([crf_group, other_group])
    model.train()
    evaluate(param=p, model=model)
    eval_res = {}
    total_step = p.train_epochs * len(train_loader)
    step = 0
    biggest_f1 = 0
    ckpt_path = None
    for i in range(p.train_epochs):
        for batch_data in tqdm(train_loader,desc=f"epoch: {i} training..."):
            input_ids, label_ids, attention_mask = map(lambda x: x.to(device), batch_data)
            # print(batch_data)
            output = model(input_ids, label_ids, attention_mask)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            step += 1
            if step % p.eval_step_intervals == 0:
                cur_eval_res = evaluate(p, model)
                if cur_eval_res["f1"] > biggest_f1:
                    biggest_f1 = cur_eval_res["f1"]
                    if ckpt_path is not None:
                        os.remove(ckpt_path)
                    ckpt_path = f"./ckpts/time{time_tuple[1]}_{time_tuple[2]}_{time_tuple[3]}_f1_{biggest_f1}.pkl"
                    torch.save(model.state_dict(), ckpt_path)
                eval_res[f"epoch{i}_step{step}_time{time_tuple[1]}_{time_tuple[2]}_{time_tuple[3]}"] = evaluate(p, model)
    print(eval_res)

def evaluate(param, model):
    test_dataset = load_dataset(param.test_file, param)
    build_features_new = partial(build_features, param=param)
    with ProcessPoolExecutor(param.eval_kernel) as pool:
        test_data = list(pool.map(build_features_new,test_dataset))
    input_ids, label_ids, attention_mask = list(zip(*test_data))
    test_data = NERDataset(input_ids, label_ids, attention_mask)
    # train_data = build_features(x, p)
    # print(feature_map)
    train_loader = DataLoader(dataset=test_data,batch_size=param.batch_size,shuffle=True,collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    # optimizer = torch.optim.Adam([crf_group, other_group])
    model.eval()
    y_true = []
    y_pred = []
    for batch_data in tqdm(train_loader,desc=f"Evaluating..."):
        input_ids, label_ids, attention_mask = map(lambda x: x.to(device), batch_data)
        # print(batch_data)
        cur_step_y_pred = model(input_ids, attention_mask=attention_mask)
        y_pred.extend(cur_step_y_pred)
        for i,y in enumerate(label_ids.tolist()):
            y_true.append([param.label_map_reverse[t] for t in y[1:len(cur_step_y_pred[i]) + 1]])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true,y_pred)
    print("f1:",f1)
    return {
        "acc" : acc,
        "f1"  : f1
    }

if __name__ == "__main__":
    # y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    # y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    # f1 = f1_score(y_true, y_pred)
    # a = accuracy_score(y_true, y_pred)
    # b = classification_report(y_true, y_pred)
    # print(f1, a, b)
    # evaluate()
    main()
    # p = Param()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x = load_dataset(p.test_file, p)
    # train_data = build_features(x, p)
    # # print(feature_map)
    # train_loader = DataLoader(dataset=train_data,batch_size=p.batch_size,shuffle=True,collate_fn=collate_fn)
    # for batch_data in tqdm(train_loader):
    #     print((batch_data))
    #     break
    # input_ids = torch.LongTensor([[ 101, 2769, 1373, 6627, 3173, 3696, 2769,  738, 6206, 1391, 3165, 1157,
    #      1157, 1962, 1450, 5489, 1259, 5291, 8043, 6821, 6573, 4994, 4197, 3221,
    #      5489, 1259, 5291, 1922, 5846,  749, 1416,  840, 2157, 6963,  679, 5650,
    #      2533, 1391, 1568, 1599, 3614, 1391, 8024, 2218, 1068, 3800, 1391, 6573,
    #      6963, 1762, 6821, 7027, 1256, 3022, 1416,    0,    0,    0,    0,    0,
    #         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #         0,    0,    0,    0,    0,    0,    0,  102]])
    # attention_mask = torch.LongTensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #      1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0.]])
    # label_ids = torch.LongTensor([[0,    0,    0,    1,    2,    2,    0,    0,    0,    0,    0,    0,
    #         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    2,
    #         0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, -100,
    #      -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
    #      -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
    #      -100, -100, -100, -100, -100, -100, -100, -100]])
    
    
    
    # print(attention_mask[:,0])
    # model = Vpn(p)
    # print(model.named_parameters())
    # print(attention_mask[0,58])
    # print(model(input_ids, attention_mask, label_ids))
    # print(p.num_tags)
    # for name, param in model.named_parameters():
    #     print(name,type(param), param.size())
