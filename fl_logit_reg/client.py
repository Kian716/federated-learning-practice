import model
import tensorflow as tf

class client:
    def __init__(self, conf, dataset) -> None:
        self.conf = conf # 读取conf

        self.model = model.Model()

        self.dataset = dataset 

        self.data_iter = tf.data.Dataset.from_tensor_slices((self.dataset[:,:-1], self.dataset[:,-1])).shuffle(buffer_size=self.dataset.shape[0]).batch(self.conf['batch_size'])
        
    def local_train(self):
        pass
