import tensorflow as tf

slim = tf.contrib.slim


def lstm_layer(input_list, num_units, forget_bias=1, name='lstm'):
    """
    :param input: X T * [B, F]
    :param num_units: C
    :param forget_bias: 1
    :param name: lstm
    :return: outputs T * [B, num_units]
             state       [B, num_units]
    """
    (B, F) = input_list[0].shape.as_list()
    initializer = tf.contrib.layers.xavier_initializer()
    wi = tf.get_variable(name + '_wi', shape=(F, num_units), initializer=initializer)
    ui = tf.get_variable(name + '_ui', shape=(num_units, num_units), initializer=initializer)
    bi = tf.get_variable(name + '_bi', shape=(1, num_units), initializer=initializer)
    wf = tf.get_variable(name + '_wf', shape=(F, num_units), initializer=initializer)
    uf = tf.get_variable(name + '_uf', shape=(num_units, num_units), initializer=initializer)
    bf = tf.get_variable(name + '_bf', shape=(1, num_units), initializer=initializer)
    wo = tf.get_variable(name + '_wo', shape=(F, num_units), initializer=initializer)
    uo = tf.get_variable(name + '_uo', shape=(num_units, num_units), initializer=initializer)
    bo = tf.get_variable(name + '_bo', shape=(1, num_units), initializer=initializer)
    wc = tf.get_variable(name + '_wc', shape=(F, num_units), initializer=initializer)
    uc = tf.get_variable(name + '_uc', shape=(num_units, num_units), initializer=initializer)
    bc = tf.get_variable(name + '_bc', shape=(1, num_units), initializer=initializer)

    def lstm_cell(x, c, h):
        """
        :param x: [B, F]
        :param c: [B, num_units]
        :param h: [B, num_units]
        :return:
        """
        it = tf.sigmoid(tf.matmul(x, wi) + tf.matmul(h, ui) + bi)
        ft = tf.sigmoid(tf.matmul(x, wf) + tf.matmul(h, uf) + bf + forget_bias)
        ot = tf.sigmoid(tf.matmul(x, wo) + tf.matmul(h, uo) + bo)
        ct = tf.tanh(tf.matmul(x, wc) + tf.matmul(h, uc) + bc)

        c_new = (ft * c) + (it * ct)
        h_new = ot * tf.tanh(c_new)

        return c_new, h_new

    def clstm_cell(x, c, h):
        """
        :param x: [B, F]
        :param c: [B, num_units]
        :param h: [B, num_units]
        :return:
        """
        it = tf.sigmoid(tf.matmul(x, wi) + tf.matmul(h, ui) + bi)
        ft = tf.sigmoid(bf + forget_bias)
        ct = tf.tanh(tf.matmul(x, wc) + tf.matmul(h, uc) + bc)

        c_new = (ft * c) + (it * ct)
        h_new = tf.tanh(c_new)

        return c_new, h_new

    outputs = []
    states = []
    state = tf.Variable(tf.zeros([B, num_units]), trainable=False, name='initial_state')
    output = tf.Variable(tf.zeros([B, num_units]), trainable=False, name='initial_output')
    for ipt in input_list:
        # state, output = lstm_cell(ipt, state, output)
        state, output = clstm_cell(ipt, state, output)
        outputs.append(output)
        states.append(state)

    return outputs, states[-1]


def bilstm_layer(input_list, num_units, forget_bias=1, name='bilstm'):
    """
    :param input: X T * [B, F]
    :param num_units: C
    :param forget_bias: 1
    :param name: bilstm
    :return: outputs T * [B, 2 * num_units]
             state       [B, 2 * num_units]
    """
    T = len(input_list)
    (B, F) = input_list[0].shape.as_list()
    w_initializer = tf.glorot_normal_initializer()
    u_initializer = tf.glorot_normal_initializer()  # tf.contrib.layers.xavier_initializer()
    b_initializer = tf.glorot_normal_initializer()  # tf.zeros_initializer()

    f_wi = tf.get_variable(name + '_f_wi', shape=(F, num_units), initializer=w_initializer)
    f_ui = tf.get_variable(name + '_f_ui', shape=(num_units, num_units), initializer=u_initializer)
    f_bi = tf.get_variable(name + '_f_bi', shape=(1, num_units), initializer=b_initializer)
    f_wf = tf.get_variable(name + '_f_wf', shape=(F, num_units), initializer=w_initializer)
    f_uf = tf.get_variable(name + '_f_uf', shape=(num_units, num_units), initializer=u_initializer)
    f_bf = tf.get_variable(name + '_f_bf', shape=(1, num_units), initializer=b_initializer)
    f_wo = tf.get_variable(name + '_f_wo', shape=(F, num_units), initializer=w_initializer)
    f_uo = tf.get_variable(name + '_f_uo', shape=(num_units, num_units), initializer=u_initializer)
    f_bo = tf.get_variable(name + '_f_bo', shape=(1, num_units), initializer=b_initializer)
    f_wc = tf.get_variable(name + '_f_wc', shape=(F, num_units), initializer=w_initializer)
    f_uc = tf.get_variable(name + '_f_uc', shape=(num_units, num_units), initializer=u_initializer)
    f_bc = tf.get_variable(name + '_f_bc', shape=(1, num_units), initializer=b_initializer)

    b_wi = tf.get_variable(name + '_b_wi', shape=(F, num_units), initializer=w_initializer)
    b_ui = tf.get_variable(name + '_b_ui', shape=(num_units, num_units), initializer=u_initializer)
    b_bi = tf.get_variable(name + '_b_bi', shape=(1, num_units), initializer=b_initializer)
    b_wf = tf.get_variable(name + '_b_wf', shape=(F, num_units), initializer=w_initializer)
    b_uf = tf.get_variable(name + '_b_uf', shape=(num_units, num_units), initializer=u_initializer)
    b_bf = tf.get_variable(name + '_b_bf', shape=(1, num_units), initializer=b_initializer)
    b_wo = tf.get_variable(name + '_b_wo', shape=(F, num_units), initializer=w_initializer)
    b_uo = tf.get_variable(name + '_b_uo', shape=(num_units, num_units), initializer=u_initializer)
    b_bo = tf.get_variable(name + '_b_bo', shape=(1, num_units), initializer=b_initializer)
    b_wc = tf.get_variable(name + '_b_wc', shape=(F, num_units), initializer=w_initializer)
    b_uc = tf.get_variable(name + '_b_uc', shape=(num_units, num_units), initializer=u_initializer)
    b_bc = tf.get_variable(name + '_b_bc', shape=(1, num_units), initializer=b_initializer)

    def forward_lstm_cell(x, f_c, f_h):
        """
        :param x: [B, F]
        :param f_c: [B, num_units]
        :param f_h: [B, num_units]
        :return:
        """
        f_it = tf.sigmoid(tf.matmul(x, f_wi) + tf.matmul(f_h, f_ui) + f_bi)
        f_ft = tf.sigmoid(tf.matmul(x, f_wf) + tf.matmul(f_h, f_uf) + f_bf + forget_bias)
        f_ot = tf.sigmoid(tf.matmul(x, f_wo) + tf.matmul(f_h, f_uo) + f_bo)
        f_ct = tf.tanh(tf.matmul(x, f_wc) + tf.matmul(f_h, f_uc) + f_bc)

        f_c_new = (f_ft * f_c) + (f_it * f_ct)
        f_h_new = f_ot * tf.tanh(f_c_new)

        return f_c_new, f_h_new

    def backward_lstm_cell(x, b_c, b_h):
        """
        :param x: [B, F]
        :param b_c: [B, num_units]
        :param b_h: [B, num_units]
        :return:
        """
        b_it = tf.sigmoid(tf.matmul(x, b_wi) + tf.matmul(b_h, b_ui) + b_bi)
        b_ft = tf.sigmoid(tf.matmul(x, b_wf) + tf.matmul(b_h, b_uf) + b_bf + forget_bias)
        b_ot = tf.sigmoid(tf.matmul(x, b_wo) + tf.matmul(b_h, b_uo) + b_bo)
        b_ct = tf.tanh(tf.matmul(x, b_wc) + tf.matmul(b_h, b_uc) + b_bc)

        b_c_new = (b_ft * b_c) + (b_it * b_ct)
        b_h_new = b_ot * tf.tanh(b_c_new)

        return b_c_new, b_h_new

    forward_outputs = []
    forward_states = []
    forward_state = tf.Variable(tf.zeros([B, num_units]), trainable=False, name='forward_initial_state')
    forward_output = tf.Variable(tf.zeros([B, num_units]), trainable=False, name='forward_initial_output')

    backward_outputs = []
    backward_states = []
    backward_state = tf.Variable(tf.zeros([B, num_units]), trainable=False, name='backward_initial_state')
    backward_output = tf.Variable(tf.zeros([B, num_units]), trainable=False, name='backward_initial_output')

    for i in range(T):
        forward_state, forward_output = forward_lstm_cell(input_list[i], forward_state, forward_output)
        forward_outputs.append(forward_output)
        forward_states.append(forward_state)
        backward_state, backward_output = backward_lstm_cell(input_list[T - 1 - i], backward_state, backward_output)
        backward_outputs.append(backward_output)
        backward_states.append(backward_state)

    outputs = []
    states = []
    for i in range(T):
        outputs.append(tf.concat([forward_outputs[i], backward_outputs[T - 1 - i]], axis=1))
        states.append(tf.concat([forward_states[i], backward_states[T - 1 - i]], axis=1))

    return outputs, states[-1]
