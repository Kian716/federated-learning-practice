import tensorflow as tf
from model import Model
import json
import dataset
import numpy as np


class Server:
    def __init__(self, conf, test_dataset) -> None:
        self.conf = conf
        self.dataset = tf.convert_to_tensor(test_dataset)
        self.model = Model(self.dataset[:, :-1].shape[1])
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def aggregate(self, clients_vars: dict) -> None:
        # 计算每个client的参数改变量
        clients_diffs = []  # [[Variable1_diff, Variable2_diff], [Variable1_diff, Variable2_diff]]
        for client_id, variables in clients_vars.items():
            diffs = []  # [Variable1_diff, Variable2_diff]
            for i, var in enumerate(variables):
                diff = var - self.model.trainable_variables[i]
                diffs.append(diff)
            clients_diffs.append(diffs)
        # 对参数变化求平均
        diff_mean = np.mean(np.array(clients_diffs, dtype=object), axis=0)
        for i, diff in enumerate(diff_mean):
            temp = tf.Variable(initial_value=self.model.trainable_variables[i])
            self.model.trainable_variables[i].assign(tf.add(temp, self.conf['lambda']*diff))
        # 将参数传回给各个clients，以server.model的形式，故无需return

    def model_eval(self):
        y = self.dataset[:, -1]
        y_pred = self.model(self.dataset[:, :-1])
        test_loss = self.loss(y_pred=y_pred, y_true=y)
        test_acc = self.evaluate(y_pred, y)
        return test_loss, test_acc

    @staticmethod
    def evaluate(y_pred, y):
        ones = tf.ones_like(y_pred)
        zeros = tf.zeros_like(y_pred)
        output = tf.where(y_pred > 0.5, ones, zeros)
        y = tf.reshape(y, output.shape)
        res = tf.reduce_sum(tf.where(y == tf.cast(output, dtype=y.dtype), ones, zeros)) / y.shape[0]
        return res.numpy()


if __name__ == '__main__':
    with open('./utils/config.json') as f:
        conf = json.load(f)
    test_dataset, train_dataset_dict = dataset.get_dataset()
    server = Server(conf, test_dataset)
    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.Dense(input_dim=30, units=1, weights=(tf.ones(shape=(30,1)),tf.ones(1))))
    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Dense(input_dim=30, units=1, weights=(tf.zeros(shape=(30,1)), tf.ones(1))))
    # print(model2.trainable_variables)
    dic = {"model1": model1.trainable_variables, "model2": model2.trainable_weights}
    # print(dic)
    # print(server.model.trainable_variables[0])
    server.aggregate(dic)
    # print(server.model.trainable_variables)
    server.model_eval()


    # clients_diffs = []
    # for client_id, variables in dic.items():
    #     diffs = []  # [Variable1_diff, Variable2_diff]
    #     for i, var in enumerate(variables):
    #         diff = var - server.model.trainable_variables[i]
    #         diffs.append(diff)
    #     # print(diffs)
    #     clients_diffs.append(diffs)
    #
    # for i, var in enumerate(np.mean(np.array(clients_diffs, dtype=object), axis=0)):
    #     temp = tf.Variable(initial_value=server.model.trainable_variables[i])
    #     server.model.trainable_variables[i].assign(tf.add(temp, conf["lambda"]*var))
    # print(server.model.trainable_variables)