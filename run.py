import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
from torch.optim import Adam
from conlleval import evaluate as ner_eval

def main():
    time_tuple = time.localtime(time.time())
    p = Param()
    print(p.__dict__)
    
    ### 加载训练数据
    x = load_dataset(p.train_file, p)
    build_features_new = partial(build_features, param=p)
    with ProcessPoolExecutor(p.train_kernel) as pool:
        train_data = list(tqdm(pool.map(build_features_new,x),total=len(x)))
    input_ids, label_ids, attention_mask = list(zip(*train_data))
    train_data = NERDataset(input_ids, label_ids, attention_mask)
    train_loader = DataLoader(dataset=train_data,batch_size=p.batch_size,
                              shuffle=True,collate_fn=partial(collate_fn,label_pad=p.label_map["O"]))
    
    ### 声明模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Vpn(p)
    # print(torch.cuda.device_count())
    model.to(device)
    # loss_fn = CrossEntropyLoss()
    
    
    ### 设定分层学习率
    crf_group = {"params":[], "lr":p.crf_lr}
    other_group = {"params":[], "lr":p.lr}
    for name, para in model.named_parameters():
        if "crf" in name:
            crf_group["params"].append(para)
            # print("crf:", name, para)
        else:
            other_group["params"].append(para)
    optimizer = torch.optim.Adam([crf_group, other_group])
    # print(optimizer)
    # print(model.optimizer.state_dict()['param_groups'][0]['lr'])

    # 训练前先评测一下
    evaluate(param=p, model=model)
    
    eval_res = {}
    total_step = p.train_epochs * len(train_loader)
    step = 0
    biggest_f1 = 0
    ckpt_path = None
    
    model.train()
    for i in range(p.train_epochs):
        for batch_data in tqdm(train_loader,desc=f"epoch: {i} training..."):
            cur_epoch_time = time.time()
            # print(batch_data)
            input_ids, label_ids, attention_mask = map(lambda x: x.to(device), batch_data)
            # print(batch_data)
            # sdfa
            output = model(input_ids, label_ids, attention_mask)
            print("loss:", output)
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
                    ckpt_path = f"./ckpts/{p.dataset_name}/top_k_{p.top_k}_{p.sum_op}_time{time_tuple[1]}_{time_tuple[2]}_{time_tuple[3]}_f1_{biggest_f1}.pkl"
                    torch.save(model.state_dict(), ckpt_path)
                eval_res[f"epoch{i}_step{step}_time{time_tuple[1]}_{time_tuple[2]}_{time_tuple[3]}"] = evaluate(p, model)
            cur_step = time.time() - cur_epoch_time
            remain_time = cur_step * (total_step - 1 - step)
        print("第{}个epoch训练结束,当前epoch最大的F1值为: {}, 剩余训练时间约为{}时{}分".format(i,biggest_f1,(remain_time / 3600), (remain_time % 3600) / 60))
    print(eval_res)
    print("biggest_f1:",biggest_f1)

def evaluate(param, model):
    print("正在进行测试...")
    test_dataset = load_dataset(param.test_file, param)
    build_features_new = partial(build_features, param=param)
    with ProcessPoolExecutor(param.eval_kernel) as pool:
        test_data = list(pool.map(build_features_new,test_dataset))
    input_ids, label_ids, attention_mask = list(zip(*test_data))
    test_data = NERDataset(input_ids, label_ids, attention_mask)
    # train_data = build_features(x, p)
    # print(feature_map)
    train_loader = DataLoader(dataset=test_data,batch_size=param.eval_batch_size,shuffle=False,collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    # optimizer = torch.optim.Adam([crf_group, other_group])
    model.eval()
    y_true = []
    y_pred = []
    la = True
    for batch_data in train_loader:
        input_ids, label_ids, attention_mask = map(lambda x: x.to(device), batch_data)
        # print(batch_data)
        cur_step_y_pred = model(input_ids, attention_mask=attention_mask)
        length = [len(cur) for cur in cur_step_y_pred]
        for i, y in enumerate(label_ids.tolist()):
            y_true.append([param.label_map_reverse[t] for t in y[:length[i]]])
            y_pred.append([param.label_map_reverse[t] for t in cur_step_y_pred[i]])
    
    # print("y_true:",y_true[:10])
    # print("y_pred:",y_pred[:10])
    # print("y_true:",[len(y) for y in y_true[:10]])
    # print("y_pred:",[len(y) for y in y_pred[:10]])
    res = ner_eval(sum(y_true,[]), sum(y_pred,[]))
    # f1 = ner_eval(sum(y_true,[]), sum(y_pred,[]))

    print("测试的结果f1:",res[2])
    return {
        "precision":res[0],
        "recall" : res[1],
        "f1"  : res[2]
    }

if __name__ == "__main__":
    # y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    # y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    # f1 = f1_score(y_true, y_pred)
    # a = accuracy_score(y_true, y_pred)
    # b = classification_report(y_true, y_pred)
    # print(f1, a, b)
    # evaluate()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print("开始时间",time.localtime())
    torch.manual_seed(3407)
    print("PID:", os.getpid())
    main()
    print("结束时间", time.localtime())
    
    # p = Param()
    # # print(p.label_map_reverse)
    # # print(p.label_map)
    # model = Vpn(p)
    # # model.load_state_dict(torch.load('/home/liuhq/hun/my_work/prompt_ner_torch/ckpts/weibo/time3_12_2_f1_0.7000000000000001.pkl'))
    # # for name, para in model.named_parameters():
    # #     if "crf" in name:
    # #         print("crf:", name, para)
    # #     else:
    # #         pass
    #         # other_group["params"].append(para)
    # print(evaluate(p, model))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x = load_dataset(p.test_file, p)
    # train_data = build_features(x, p)
    # # print(feature_map)
    # train_loader = DataLoader(dataset=train_data,batch_size=p.batch_size,shuffle=True,collate_fn=collate_fn)
    # for batch_data in tqdm(train_loader):
    #     print((batch_data))
    #     break
    # print(attention_mask[:,0])
    # for name, param in model.named_parameters():
    #     if "crf" in name:
    #         print(name, param)

    # print(model.named_parameters())
    # print(attention_mask[0,58])
    # print(model(input_ids, attention_mask, label_ids))
    # print(p.num_tags)
    # for name, param in model.named_parameters():
    #     print(name,type(param), param.size())
    
