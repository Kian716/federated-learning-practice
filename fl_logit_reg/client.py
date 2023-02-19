from model import Model
import tensorflow as tf
import json
import dataset

class Client:
    def __init__(self, id, conf, dataset) -> None:
        self.id = id
        self.conf = conf
        self.dataset = tf.convert_to_tensor(dataset)
        self.local_model = Model(self.dataset[:, :-1].shape[1])
        self.data_iter = tf.data.Dataset.from_tensor_slices((self.dataset[:, :-1], self.dataset[:, -1])).\
            shuffle(buffer_size=dataset.shape[0]).batch(self.conf['batch_size'])
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.conf['lr'])

    def train(self, global_model: Model) -> None:
        # 使用sever端传过来的参数
        for i, var in enumerate(global_model.trainable_variables):
            self.local_model.trainable_variables[i].assign(var)
        for epoch in range(self.conf['num_epochs']):
            for X, y in self.data_iter:
                with tf.GradientTape() as tape:
                    y_pred = self.local_model(X)
                    loss = self.loss(y_pred=y_pred, y_true=y)
                grads = tape.gradient(loss, self.local_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.local_model.trainable_variables))
            train_loss = self.loss(y_pred=self.local_model(X), y_true=y)
            train_acc = self.evaluate(self.local_model(X), y)
            print("epoch:{:d}, train_loss:{:.4f}, train_acc:{:.2f}".format(epoch+1, train_loss, train_acc))
        # 将训练后的参数上传给server
        return self.local_model.trainable_variables

    @staticmethod
    def evaluate(y_pred, y):
        ones = tf.ones_like(y_pred)
        zeros = tf.zeros_like(y_pred)
        output = tf.where(y_pred > 0.5, ones, zeros)
        y = tf.reshape(y, output.shape)
        res = tf.reduce_sum(tf.where(y == tf.cast(output, dtype=y.dtype), ones, zeros)) / y.shape[0]
        return res.numpy()

if __name__ == "__main__":
    with open('./utils/config.json') as f:
        conf = json.load(f)
    test_dataset,train_dataset_dict = dataset.get_dataset()
    client = Client(1, conf, train_dataset_dict['A'])
    # for X, y in client.data_iter:
    #     print(X, y)
    #     break
    global_model = Model(30)
    client.train(global_model)
        