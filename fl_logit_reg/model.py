import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义模型并初始化参数
class Model(tf.keras.Sequential):
    '''logistic regression model'''
    def __init__(self, input_size) -> None:
        super().__init__()
        self.initializer = tf.keras.initializers.Zeros()
        self.add(tf.keras.layers.Dense(input_dim=input_size, units=1, kernel_initializer=self.initializer,
                                       activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L1(0.02)))


if __name__ == '__main__':
    model = Model(3)
    X = tf.constant([[1,2,3],[1,2,3]])
    print(model(X))
    print(model.trainable_weights)
    
