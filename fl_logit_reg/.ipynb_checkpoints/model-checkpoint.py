
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 定义模型并初始化参数
class Model(tf.keras.Model):
    '''logistic regression model'''
    def __init__(self) -> None:
        super().__init__()
        self.model = tf.keras.Sequential()
        self.initializer = tf.keras.initializers.Zeros()
        self.model.add(tf.keras.layers.Dense(1, kernel_initializer=self.initializer, activation='sigmoid'))

    def __call__(self, X: tf.Tensor, **kwds) -> tf.Tensor:
        return self.model(X)

if __name__ == '__main__':
    model = Model()
    X = tf.constant([[1,2,3],[1,2,3]])
    print(model(X))
    print(model.trainable_weights)
    
