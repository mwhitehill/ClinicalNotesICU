import tensorflow as tf

def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, bnorm, scope):
    assert bnorm in ('before', 'after')
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation if bnorm == 'after' else None,
            padding='same')
        batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
        activated = activation(batched) if bnorm == 'before' else batched
        return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
                                name='dropout_{}'.format(scope))

class EncoderConvolutions:
    """Encoder convolutional layers used to find local dependencies in inputs characters.
    """
    def __init__(self, is_training, activation=tf.nn.relu, scope=None):
        """
        Args:
            is_training: Boolean, determines if the model is training or in inference to control dropout
            kernel_size: tuple or integer, The size of convolution kernels
            channels: integer, number of convolutional kernels
            activation: callable, postnet activation function for each convolutional layer
            scope: Postnet scope.
        """
        super(EncoderConvolutions, self).__init__()
        self.is_training = is_training

        self.kernel_size = (5, )
        self.channels = 512
        self.activation = activation
        self.scope = 'enc_conv_layers' if scope is None else scope
        self.drop_rate = .5
        self.enc_conv_num_layers = 3
        self.bnorm = 'after'

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.enc_conv_num_layers):
                x = conv1d(x, self.kernel_size, self.channels, self.activation,
                    self.is_training, self.drop_rate, self.bnorm, 'conv_layer_{}_'.format(i + 1)+self.scope)
        return x

class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Wrapper for tf LSTM to create Zoneout LSTM Cell
    inspired by:
    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py
    Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.
    Many thanks to @Ondal90 for pointing this out. You sir are a hero!
    '''
    def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
        '''Initializer with possibility to set different zoneout values for cell/hidden states.
        '''
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        '''Runs vanilla LSTM Cell and applies zoneout.
        '''
        #Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

        #Apply zoneout
        if self.is_training:
            #nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

        return output, new_state

class EncoderRNN:
    """Encoder bidirectional one layer LSTM
    """
    def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
        """
        Args:
            is_training: Boolean, determines if the model is training or in inference to control zoneout
            size: integer, the number of LSTM units for each direction
            zoneout: the zoneout factor
            scope: EncoderRNN scope.
        """
        super(EncoderRNN, self).__init__()
        self.is_training = is_training

        self.size = size
        self.zoneout = zoneout
        self.scope = 'encoder_LSTM' if scope is None else scope

        #Create forward LSTM Cell
        self._fw_cell = ZoneoutLSTMCell(size*2, is_training,
            zoneout_factor_cell=zoneout,
            zoneout_factor_output=zoneout,
            name='encoder_fw_LSTM')

        # #Create backward LSTM Cell
        # self._bw_cell = ZoneoutLSTMCell(size, is_training,
        #     zoneout_factor_cell=zoneout,
        #     zoneout_factor_output=zoneout,
        #     name='encoder_bw_LSTM')

    def __call__(self, inputs, input_lengths):
        with tf.variable_scope(self.scope):
            # outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            #     self._fw_cell,
            #     self._bw_cell,
            #     inputs,
            #     sequence_length=input_lengths,
            #     dtype=tf.float32,
            #     swap_memory=True)
            # outputs = tf.concat(outputs, axis=2)

            outputs, (fw_state, bw_state) = tf.nn.dynamic_rnn(
                self._fw_cell,
                inputs,
                sequence_length=input_lengths,
                dtype=tf.float32,
                swap_memory=True)

            return  outputs # Concat and return forward + backward outputs

class TacotronEncoderCell(tf.nn.rnn_cell.RNNCell):
    """Tacotron 2 Encoder Cell
    Passes inputs through a stack of convolutional layers then through a bidirectional LSTM
    layer to predict the hidden representation vector (or memory)
    """

    def __init__(self, convolutional_layers, lstm_layer):
        """Initialize encoder parameters
        Args:
            convolutional_layers: Encoder convolutional block class
            lstm_layer: encoder bidirectional lstm layer class
        """
        super(TacotronEncoderCell, self).__init__()
        #Initialize encoder layers
        self._convolutions = convolutional_layers
        self._cell = lstm_layer

    def __call__(self, inputs, input_lengths=None):
        #Pass input sequence through a stack of convolutional layers
        conv_output = self._convolutions(inputs)

        #Extract hidden representation from encoder lstm cells
        hidden_representation = self._cell(conv_output, input_lengths)

        #For shape visualization
        self.conv_output_shape = conv_output.shape
        return hidden_representation



