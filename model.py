
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Input, Layer, BatchNormalization, LayerNormalization, ReLU, Dropout
from tensorflow.keras.regularizers import l2

class CrossNetwork(Layer):
    def __init__(self, layer_num, reg_w=0., reg_b=0.):
        """CrossNetwork

        Args:
            layer_num (scalar): The depth of cross network
            reg_w (scalar, optional): Regularization of w. Defaults to 0.
            reg_b (int, optional): regularization of b. Defaults to 0.
        """
        super(CrossNetwork, self).__init__()
        self._layer_num = layer_num
        self._reg_w = reg_w
        self._reg_b = reg_b
        
    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weight = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer="random_normal",
                            regularizer=l2(self._reg_w),
                            trainable=True)
            for i in range(self._layer_num)
        ]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer="random_normal",
                            regularizer=l2(self._reg_b),
                            trainable=True)
            for i in range(self._layer_num)
        ]
    
    def call(self, inputs):
        x_0 = tf.expand_dims(inputs, axis=2) # (batch_size, dim, 1)
        x_l = x_0 # (None, dim, 1)
        for i in range(self._layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weight[i], axes=[1, 0]) # (batch_size, 1, 1)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l # (batch_sizt, dim, 1)
        x_l = tf.squeeze(x_l, axis=2) # (bth, dim)
        return x_l
    

class MLP(Layer):
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., use_batch_norm=False):
        """Multilayer Perceptron.
        Args:
            :param hidden_units: A list. The list of hidden layer units's numbers.
            :param activation: A string. The name of activation function, like 'relu', 'sigmoid' and so on.
            :param dnn_dropout: A scalar. The rate of dropout .
            :param use_batch_norm: A boolean. Whether using batch normalization or not.
        :return:
        """
        super(MLP, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.use_batch_norm = use_batch_norm
        self.bt = BatchNormalization()

    def call(self, inputs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        if self.use_batch_norm:
            x = self.bt(x)
        x = self.dropout(x)
        return x
        

class DCN(Model):
    def __init__(self, feature_columns, hidden_units, activation="relu",
                 dnn_dropout=0., embed_reg=0., cross_w_reg=0., cross_b_reg=0.):
        """Deep&Cross Network

        Args:
            feature_columns (list<dict>): [{feat_name:, feat_num:, embed_dim:}, ...].
            hidden_units (list): such as [128, 64].
            activation (str, optional): Activation function of MLP Defaults to 'relu'.
            dnn_dropout (scalar, optional):dropout rate of MLP. Defaults to 0..
            embed_reg (scalar, optional): The regularization coefficient of embedding. Defaults to 0..
            cross_w_reg (scalar, optional): The regularization coefficient of cross network. Defaults to 0..
            cross_b_reg (scalar, optional): The reguarization coefficient of cross network. Defaults to 0..
        """
        
        super(DCN, self).__init__()
        self._feature_columns = feature_columns
        self._layer_num = len(hidden_units)
        self.embed_layers = {
            feat["feat_name"]: Embedding(input_dim=feat["feat_num"],
                                         input_length=1,
                                         output_dim=feat["embed_dim"],
                                         embeddings_initializer="random_normal",
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self._feature_columns
        }
        self.cross_network = CrossNetwork(self._layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = MLP(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1, activation=None)
        
    def call(self, inputs):
        # embedding,  (batch_size, embed_dim * fields)
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        x = sparse_embed
        # Cross Network
        cross_x = self.cross_network(x)
        # DNN
        dnn_x = self.dnn_network(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self._feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()
        
        

if __name__ == "__main__":
    model_params = {
        'hidden_units': [256, 128, 64],
        'dnn_dropout': 0.5,
        'embed_reg': 0.,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.
    }
    feature_columns = [{"feat_name": 'a', "feat_num": 16, "embed_dim": 8}, {"feat_name": 'b', "feat_num": 14, "embed_dim": 7}]
    model = DCN(feature_columns, **model_params)
    model.summary()
    