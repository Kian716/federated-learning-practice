import model
import tensorflow as tf

class Client:
    def __init__(self) -> None:
        # self.conf = conf # 读取conf

        self.model = model.Model()

        # self.dataset = dataset 

        # self.data_iter = tf.data.Dataset.from_tensor_slices((self.dataset[:,:-1], self.dataset[:,-1])).shuffle(buffer_size=self.dataset.shape[0]).batch(self.conf['batch_size'])
        
    def local_train(self):
        # 获取服务端传过来的模型参数
        print(self.model.trainable_variables)

if __name__ == '__main__':
    client = Client()
    print('yes')
    client.local_train()
        
        