import json
import dataset
from server import Server
from client import Client

if __name__ == "__main__":
    with open('./utils/config.json') as f:
        conf = json.load(f)

    test_dataset, train_dataset_dict = dataset.get_dataset()
    server = Server(conf, test_dataset)
    clients = {}
    for client_name, train_dataset in train_dataset_dict.items():
        clients[client_name] = Client(client_name, conf, train_dataset)

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
