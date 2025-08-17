import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

import LoadData
from models import TextCNNKAN


def get_time_dif(start_time):
    """获取时间差"""
    end_time = time.time()
    time_dif = end_time - start_time
    # 将时间差转换为友好的格式
    minutes = int(time_dif / 60)
    seconds = int(time_dif % 60)
    return f"{minutes}m {seconds}s"


def train(config, model, train_iter, dev_iter, test_iter):
    # print("============= " + config.model_name + " 模型训练&测试 =============")
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step()  # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)

            model.zero_grad()
            # 计算交叉熵损失
            loss = F.cross_entropy(outputs, labels)

            # # 添加 L2 正则化项
            # l2_reg = torch.tensor(0., device=config.device)  # 初始化为0
            # for param in model.parameters():
            #     l2_reg += torch.norm(param, 2)  # 计算参数的L2范数并累加
            # loss += config.weight_decay * l2_reg  # 将L2正则化项添加到损失中

            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                # print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        scheduler.step()  # 学习率衰减
        if flag:
            break
    writer.close()
    # output(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_auc, test_f1, test_mcc, test_loss, test_report, test_confusion, test_precision, test_recall = evaluate(
        config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    return test_acc, test_auc, test_precision, test_recall, test_f1, test_mcc


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    auc = metrics.roc_auc_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all)
    mcc = metrics.matthews_corrcoef(labels_all, predict_all)
    precision = metrics.precision_score(labels_all, predict_all, average='weighted')
    recall = metrics.recall_score(labels_all, predict_all, average='weighted')

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, auc, f1, mcc, loss_total / len(data_iter), report, confusion, precision, recall
    return acc, loss_total / len(data_iter)


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x.to(self.device), seq_len.to(self.device)), y.to(self.device)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


if __name__ == "__main__":
    # print("开始运行.....")
    class Config():
        def __init__(self):
            self.vocab_path = '../data/TLND/vocab.pkl'
            self.train_path = '../data/TLND/train.csv'
            self.dev_path = '../data/TLND/dev.csv'
            self.test_path = '../data/TLND/test.csv'
            self.pad_size = 256
            self.batch_size = 16  # 设置批次大小
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备为 GPU 或 CPU


    TextCNNKANConfig = TextCNNKAN.Config(dataset='../data', embedding='glove.6B.200d.npz')

    learningRateList = [0.001, 0.001]
    batchSizeList = [16, 16]
    padSizeList = [256, 256]
    numEpochsList = [10, 10]

    # print(result_list)
    print("================= TextCNNKAN 超参数调优实验 =================")

    results = []

    for i in range(0, len(learningRateList)):
        for j in range(0, len(batchSizeList)):
            for k in range(0, len(padSizeList)):
                for l in range(0, len(numEpochsList)):
                    config = Config()

                    config.batch_size = batchSizeList[j]
                    config.pad_size = padSizeList[k]
                    vocab, train_data, dev_data, test_data = LoadData.build_dataset(config)

                    train_iter = build_iterator(train_data, config)
                    dev_iter = build_iterator(dev_data, config)
                    test_iter = build_iterator(test_data, config)
                    result_list = []
                    print("当前参数为: ")
                    message = "学习率: " + str(learningRateList[i]) + " epoch数: " + str(
                        batchSizeList[j]) + " 每句话处理成的长度: " + str(padSizeList[k]) + " mini-batch大小: " + str(
                        numEpochsList[l])
                    print(message)
                    TextCNNKANConfig.learning_rate = learningRateList[i]
                    TextCNNKANConfig.batch_size = batchSizeList[j]
                    TextCNNKANConfig.pad_size = padSizeList[k]
                    TextCNNKANConfig.num_epochs = numEpochsList[l]
                    TextCNNKANModel = TextCNNKAN.Model(TextCNNKANConfig).to(config.device)
                    train(TextCNNKANConfig, TextCNNKANModel, train_iter, dev_iter, test_iter)
                    # test
                    test_acc, test_auc, test_precision, test_recall, test_f1, test_mcc = test(TextCNNKANConfig,
                                                                                              TextCNNKANModel,
                                                                                              test_iter)

                    results.append({
                        'learning_rate': TextCNNKANConfig.learning_rate,
                        'batch_size': TextCNNKANConfig.batch_size,
                        'pad_size': TextCNNKANConfig.pad_size,
                        'num_epochs': TextCNNKANConfig.num_epochs,
                        'test_acc': round(test_acc, 4),
                        'test_auc': round(test_auc, 4),
                        # 'test_precision': round(test_precision, 4),
                        # 'test_recall': round(test_recall, 4),
                        'test_f1': round(test_f1, 4),
                        'test_mcc': round(test_mcc, 4)
                    })

    # Save results to a CSV file
    with open('models/result/new/TextCNNKAN_grid_search_results.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Find the best result based on test mcc
    best_result = max(results, key=lambda x: x['test_mcc'])
    print("Best parameters found:")
    print(best_result)
