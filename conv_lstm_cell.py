#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging


class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):

    """
    @param num_units: 状態のチャンネル数
    @param img_size: 画像のサイズ
    @param kernel_size: 畳み込みのフィルタサイズ
    @param stride: 畳み込みのストライド
    @param user_peepholes: ピープホールを使用するかどうか
    @param cell_clip: セルの出力のクリッピング値
    @param initializer: パラメータの初期化関数
    @param forget_bias: 学習初期に忘却ゲートが閉じないためのバイアス
    @param state_is_tuple: データの受け渡しをtupleでやるかconcatでやるか
    @param activation: 発火関数
    """

    def __init__(self, num_units, input_size=None, 
                img_size=[128, 128], kernel_size=[20,20] ,stride=[1,5,5,1], use_peepholes=False, cell_clip=None,
                initializer=None, forget_bias=1.0, 
                state_is_tuple=False, activation=tanh):

        #concatは非推奨
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                        "deprecated.  Use state_is_tuple=True.", self)
        """
        普通、入力のサイズは__call__で判定するのでコンストラクタではいらない。
        zero_stateに必要なため、ここではimg_sizeは取得している。
        """
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated." % self)

        self._num_units = num_units
        self._output_size = num_units
        self._width = img_size[0]
        self._height = img_size[1]
        self._kernel_size = kernel_size
        self._stride=stride
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

        """
        中間状態と出力の２種類の値が入ってくるので2倍の数値になっている。

        """
        self._state_size = (tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
                            if self._state_is_tuple else 2 * self._num_units)

        

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    #状態の初期値。出力と状態の二つがあるのでチャンネル数は2倍
    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self._width, self._height, self._num_units*2])

    def __call__(self, inputs, state, scope=None):

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev, m_prev = array_ops.split(3, 2, state)


        with vs.variable_scope(scope or "conv_lstm_cell",
                             initializer = None) as unit_scope:

            """
            入力ゲート, 入力候補値, 忘却ゲート, 出力ゲートの値を計算。
            ４種類の重みが出てくるが、それぞれ別に計算するのではなく、まとめて計算して分割している。
            """
            conv_lstm_matrix = _conv([inputs, m_prev], 4 * self._num_units, self._kernel_size, self._stride)
            i, j, f, o = array_ops.split(3, 4, conv_lstm_matrix)


            #ピープホールに用いるフィルタ。行列の掛け算ではなくアダマール積で計算。
            if self._use_peepholes:
                w_f_diag = vs.get_variable(
                    "w_f_diag", shape=[self._width, self._height, self._num_units], dtype=tf.float32)
                w_i_diag = vs.get_variable(
                    "w_i_diag", shape=[self._width, self._height, self._num_units],dtype=tf.float32)
                w_o_diag = vs.get_variable(
                    "w_o_diag", shape=[self._width, self._height, self._num_units],dtype=tf.float32)

            #新しい状態の計算
            if self._use_peepholes:
                c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                    sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                    self._activation(j))

            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            #新しい出力の計算
            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)


        new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, m) if self._state_is_tuple
                    else array_ops.concat(3, [c, m]))
        return m, new_state

"""
各種ゲートの計算。畳み込みで行う。
@param args: 新しい入力と前の層の出力の配列。入力データを二種類にしたいときなどはここに追加する。
@param output_size: 出力チャンネル数。全ゲート分計算するために、中間状態の4倍の値を渡す。
@param kernel_size: 畳み込みフィルタのカーネルサイズ
@param stride: 畳み込みのストライド
@param bias: 畳み込み層にバイアスを加えるかどうか
@param bias: 畳み込み層のバイアスの初期値
"""
def _conv(args, output_size, kernel_size, stride, bias=True, bias_start=0.0):

    #argsがちゃんと指定されているかチェック。配列じゃない場合は囲って配列にしている。
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    #前の時間の出力画像と入力画像のチャンネルの合計値を求める
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        #前の出力、入力データはどちらも（batch, width, height, channel）の4階テンソル
        if len(shape) != 4:
            raise ValueError("conv is expecting 4D arguments: %s" % shapes)
        #batchサイズ以外が入っていないのは認めない
        if not shape[1] or not shape[2] or not shape[3]:
            raise ValueError("conv expects shape[1-3] to be provided for shape %s." % shape)
        else:
            total_arg_size += shape[3]

    dtype = [a.dtype for a in args][0]
    #__call__で定義したvariable_scopeを取得
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        #カーネルの変数を定義
        kernel = vs.get_variable(
            "kernel", [kernel_size[0], kernel_size[1], total_arg_size, output_size])
       
        # 共有重みになるので必ずpadding='SAME'で畳み込み
        if len(args) == 1:
            res = tf.nn.conv2d(args[0],kernel, stride, padding='SAME')
        else:
            res = tf.nn.conv2d(array_ops.concat(3, args), kernel, stride, padding='SAME')

        #biasのadd
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                "biases",[output_size],
                dtype = dtype,
                initializer = init_ops.constant_initializer(bias_start, dtype=dtype))
    return res + biases