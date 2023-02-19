import json
import dataset
from server import Server
from client import Client
import pandas as pd

if __name__ == "__main__":
    with open('./utils/config.json') as f:
        conf = json.load(f)

    '''联邦训练'''
    test_dataset, train_dataset_dict = dataset.get_dataset()
    server = Server(conf, test_dataset)
    clients = {}
    for client_name, train_dataset in train_dataset_dict.items():
        clients[client_name] = Client(client_name, conf, train_dataset)
    with open('./results/res_fl.txt', 'w') as f:
        for epoch in range(conf["global_epochs"]):
            # clients进行训练
            results = {}
            for client_name, client in clients.items():
                results[client_name] = client.train(server.model)
            # server进行参数聚合
            server.aggregate(results)
            # server端进行评估
            test_loss, test_acc = server.model_eval()
            print("global_epoch:{:d}, test_loss:{:.4f}, test_acc:{:.2f}".format(epoch+1, test_loss, test_acc))
            f.write("{:d},{:.4f},{:.2f}\n".format(epoch+1, test_loss, test_acc))

    '''集中式训练'''
    train_dataset_center_dict = {}
    test_dataset, train_dataset_dict = dataset.get_dataset()
    train_dataset_center_dict['center'] = pd.concat(train_dataset_dict.values(), axis=0)
    server = Server(conf, test_dataset)
    clients = {}
    for client_name, train_dataset in train_dataset_center_dict.items():
        clients[client_name] = Client(client_name, conf, train_dataset)

    with open ('./results/res_center.txt', 'w') as f:
        for epoch in range(conf["global_epochs"]):
            # clients进行训练
            results = {}
            for client_name, client in clients.items():
                results[client_name] = client.train(server.model)
            # server进行参数聚合
            server.aggregate(results)
            # server端进行评估
            test_loss, test_acc = server.model_eval()
            print("global_epoch:{:d}, test_loss:{:.4f}, test_acc:{:.2f}".format(epoch+1, test_loss, test_acc))
            f.write("{:d},{:.4f},{:.2f}\n".format(epoch+1, test_loss, test_acc))


'''
1. 关于为什么每个global_epoch，client都在本地训练epochs次，local_model没有过拟合：
是因为每一个global_epoch，client端的local_model在开始训练前，
其参数都与global_model进行了同步，
就算在之前的local_training中拟合得很好，
最后还是会返回global_model未充分拟合的状态
（global_model由于lambda超参数的存在，使得其模型拟合得很慢）

2. 关于对比集中式训练与联邦学习时，只需要将client的数量设置为1即可，
在本代码中，就是将长度为2的train_dataset_dict，改为长度为1的train_dataset_center_dict
'''