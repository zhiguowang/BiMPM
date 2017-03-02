import tensorflow as tf
from tensorflow.python.ops import rnn
import my_rnn

eps = 1e-6
def cosine_distance(y1,y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
#     cosine_numerator = T.sum(y1*y2, axis=-1)
    cosine_numerator = tf.reduce_sum(tf.mul(y1, y2), axis=-1)
#     y1_norm = T.sqrt(T.maximum(T.sum(T.sqr(y1), axis=-1), eps)) #be careful while using T.sqrt(), like in the cases of Euclidean distance, cosine similarity, for the gradient of T.sqrt() at 0 is undefined, we should add an Eps or use T.maximum(original, eps) in the sqrt.
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps)) 
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps)) 
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.mul(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.mul(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def cal_cosine_weighted_question_representation(question_representation, cosine_matrix, normalize=False):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    if normalize: cosine_matrix = tf.nn.softmax(cosine_matrix)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1) # [batch_size, passage_len, question_len, 'x']
    weighted_question_words = tf.expand_dims(question_representation, axis=1) # [batch_size, 'x', question_len, dim]
    weighted_question_words = tf.reduce_sum(tf.mul(weighted_question_words, expanded_cosine_matrix), axis=2)# [batch_size, passage_len, dim]
    if not normalize:
        weighted_question_words = tf.div(weighted_question_words, tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),eps),axis=-1))
    return weighted_question_words # [batch_size, passage_len, dim]

def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=2) #[batch_size, passage_len, 'x', dim]
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0) # [1, 1, decompse_dim, dim]
    return tf.mul(in_tensor, decompose_params)#[batch_size, passage_len, decompse_dim, dim]

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
    return tf.mul(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]

def multi_perspective_expand_for_1D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=0) #['x', dim]
    return tf.mul(in_tensor, decompose_params) # [decompse_dim, dim]


def cal_full_matching_bak(passage_representation, full_question_representation, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # full_question_representation: [batch_size, dim]
    # decompose_params: [decompose_dim, dim]
    mp_passage_rep = multi_perspective_expand_for_3D(passage_representation, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
    mp_full_question_rep = multi_perspective_expand_for_2D(full_question_representation, decompose_params) # [batch_size, decompse_dim, dim]
    return cosine_distance(mp_passage_rep, tf.expand_dims(mp_full_question_rep, axis=1)) #[batch_size, passage_len, decompse_dim]

def cal_full_matching(passage_representation, full_question_representation, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # full_question_representation: [batch_size, dim]
    # decompose_params: [decompose_dim, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_1D(q, decompose_params) # [decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, decompose_dim, dim]
        return cosine_distance(p, q) # [passage_len, decompose]
    elems = (passage_representation, full_question_representation)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]
    
def cal_maxpooling_matching_bak(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    passage_rep = multi_perspective_expand_for_3D(passage_rep, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
    question_rep = multi_perspective_expand_for_3D(question_rep, decompose_params) # [batch_size, question_len, decompse_dim, dim]

    passage_rep = tf.expand_dims(passage_rep, 2) # [batch_size, passage_len, 1, decompse_dim, dim]
    question_rep = tf.expand_dims(question_rep, 1) # [batch_size, 1, question_len, decompse_dim, dim]
    matching_matrix = cosine_distance(passage_rep,question_rep) # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat(2, [tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])# [batch_size, passage_len, 2*decompse_dim]

def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1) # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, question_len, decompose_dim, dim]
        return cosine_distance(p, q) # [passage_len, question_len, decompose]
    elems = (passage_rep, question_rep)
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat(2, [tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])# [batch_size, passage_len, 2*decompse_dim]

def cal_maxpooling_matching_for_word(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
        # p: [pasasge_len, dim], q: [question_len, dim]
        def single_instance_2(y):
            # y: [dim]
            y = multi_perspective_expand_for_1D(y, decompose_params) #[decompose_dim, dim]
            y = tf.expand_dims(y, 0) # [1, decompose_dim, dim]
            matching_matrix = cosine_distance(y, q)#[question_len, decompose_dim]
            return tf.concat(0, [tf.reduce_max(matching_matrix, axis=0), tf.reduce_mean(matching_matrix, axis=0)]) #[2*decompose_dim]
        return tf.map_fn(single_instance_2, p, dtype=tf.float32) # [passage_len, 2*decompse_dim]
    elems = (passage_rep, question_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, 2*decompse_dim]


def cal_attentive_matching(passage_rep, att_question_rep, decompose_params):
    # passage_rep: [batch_size, passage_len, dim]
    # att_question_rep: [batch_size, passage_len, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [pasasge_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [pasasge_len, decompose_dim, dim]
        return cosine_distance(p, q) # [pasasge_len, decompose_dim]

    elems = (passage_rep, att_question_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]

def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

#     xdev = x - x.max()
#     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.mul(logits, mask)
    xdev = tf.sub(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.sub(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
#     return -T.sum(targets * log_predictions)
    result = tf.mul(tf.mul(truth, log_predictions), mask) # [batch_size, passage_len]
    return tf.mul(-1.0,tf.reduce_sum(result, -1)) # [batch_size]
    
def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.mul(trans, gate), tf.mul(in_val, tf.sub(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

def cal_max_question_representation(question_representation, cosine_matrix):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    question_index = tf.arg_max(cosine_matrix, 2) # [batch_size, passage_len]
    def singel_instance(x):
        q = x[0]
        c = x[1]
        return tf.gather(q, c)
    elems = (question_representation, question_index)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, dim]

def cal_linear_decomposition_representation(passage_representation, passage_lengths, cosine_matrix,is_training, 
                                            lex_decompsition_dim, dropout_rate):
    # passage_representation: [batch_size, passage_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    passage_similarity = tf.reduce_max(cosine_matrix, 2)# [batch_size, passage_len]
    similar_weights = tf.expand_dims(passage_similarity, -1) # [batch_size, passage_len, 1]
    dissimilar_weights = tf.subtract(1.0, similar_weights)
    similar_component = tf.mul(passage_representation, similar_weights)
    dissimilar_component = tf.mul(passage_representation, dissimilar_weights)
    all_component = tf.concat(2, [similar_component, dissimilar_component])
    if lex_decompsition_dim==-1:
        return all_component
    with tf.variable_scope('lex_decomposition'):
        lex_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
        lex_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
        if is_training:
            lex_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            lex_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
        lex_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_fw])
        lex_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_bw])

        (lex_features_fw, lex_features_bw), _ = rnn.bidirectional_dynamic_rnn(
                    lex_lstm_cell_fw, lex_lstm_cell_bw, all_component, dtype=tf.float32, sequence_length=passage_lengths)

        lex_features = tf.concat(2, [lex_features_fw, lex_features_bw])
    return lex_features


def match_passage_with_question(passage_context_representation_fw, passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True):

    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        fw_question_full_rep = question_context_representation_fw[:,-1,:]
        bw_question_full_rep = question_context_representation_bw[:,0,:]

        question_context_representation_fw = tf.mul(question_context_representation_fw, tf.expand_dims(question_mask,-1))
        question_context_representation_bw = tf.mul(question_context_representation_bw, tf.expand_dims(question_mask,-1))
        passage_context_representation_fw = tf.mul(passage_context_representation_fw, tf.expand_dims(mask,-1))
        passage_context_representation_bw = tf.mul(passage_context_representation_bw, tf.expand_dims(mask,-1))

        forward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_fw, passage_context_representation_fw)
        forward_relevancy_matrix = mask_relevancy_matrix(forward_relevancy_matrix, question_mask, mask)

        backward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_bw, passage_context_representation_bw)
        backward_relevancy_matrix = mask_relevancy_matrix(backward_relevancy_matrix, question_mask, mask)
        if MP_dim > 0:
            if with_full_match:
                # forward Full-Matching: passage_context_representation_fw vs question_context_representation_fw[-1]
                fw_full_decomp_params = tf.get_variable("forward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_full_match_rep = cal_full_matching(passage_context_representation_fw, fw_question_full_rep, fw_full_decomp_params)
                all_question_aware_representatins.append(fw_full_match_rep)
                dim += MP_dim

                # backward Full-Matching: passage_context_representation_bw vs question_context_representation_bw[0]
                bw_full_decomp_params = tf.get_variable("backward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_full_match_rep = cal_full_matching(passage_context_representation_bw, bw_question_full_rep, bw_full_decomp_params)
                all_question_aware_representatins.append(bw_full_match_rep)
                dim += MP_dim

            if with_maxpool_match:
                # forward Maxpooling-Matching
                fw_maxpooling_decomp_params = tf.get_variable("forward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_fw, question_context_representation_fw, fw_maxpooling_decomp_params)
                all_question_aware_representatins.append(fw_maxpooling_rep)
                dim += 2*MP_dim
                # backward Maxpooling-Matching
                bw_maxpooling_decomp_params = tf.get_variable("backward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_bw, question_context_representation_bw, bw_maxpooling_decomp_params)
                all_question_aware_representatins.append(bw_maxpooling_rep)
                dim += 2*MP_dim
            
            if with_attentive_match:
                # forward attentive-matching
                # forward weighted question representation: [batch_size, question_len, passage_len] [batch_size, question_len, context_lstm_dim]
                att_question_fw_contexts = cal_cosine_weighted_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_attentive_decomp_params = tf.get_variable("forward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_attentive_rep = cal_attentive_matching(passage_context_representation_fw, att_question_fw_contexts, fw_attentive_decomp_params)
                all_question_aware_representatins.append(fw_attentive_rep)
                dim += MP_dim

                # backward attentive-matching
                # backward weighted question representation
                att_question_bw_contexts = cal_cosine_weighted_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_attentive_decomp_params = tf.get_variable("backward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_attentive_rep = cal_attentive_matching(passage_context_representation_bw, att_question_bw_contexts, bw_attentive_decomp_params)
                all_question_aware_representatins.append(bw_attentive_rep)
                dim += MP_dim
            
            if with_max_attentive_match:
                # forward max attentive-matching
                max_att_fw = cal_max_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_max_att_decomp_params = tf.get_variable("fw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_max_attentive_rep = cal_attentive_matching(passage_context_representation_fw, max_att_fw, fw_max_att_decomp_params)
                all_question_aware_representatins.append(fw_max_attentive_rep)
                dim += MP_dim

                # backward max attentive-matching
                max_att_bw = cal_max_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_max_att_decomp_params = tf.get_variable("bw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_max_attentive_rep = cal_attentive_matching(passage_context_representation_bw, max_att_bw, bw_max_att_decomp_params)
                all_question_aware_representatins.append(bw_max_attentive_rep)
                dim += MP_dim

        all_question_aware_representatins.append(tf.reduce_max(forward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(forward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_max(backward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(backward_relevancy_matrix, axis=2,keep_dims=True))
        dim += 4
    return (all_question_aware_representatins, dim)
        
def unidirectional_matching(in_question_repres, in_passage_repres,question_lengths, passage_lengths,
                            question_mask, mask, MP_dim, input_dim, with_filter_layer, context_layer_num,
                            context_lstm_dim,is_training,dropout_rate,with_match_highway,aggregation_layer_num,
                            aggregation_lstm_dim,highway_layer_num,with_aggregation_highway,with_lex_decomposition, lex_decompsition_dim,
                            with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True):
    # ======Filter layer======
    cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres)
    cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
    raw_in_passage_repres = in_passage_repres
    if with_filter_layer:
        relevancy_matrix = cosine_matrix # [batch_size, passage_len, question_len]
        relevancy_degrees = tf.reduce_max(relevancy_matrix, axis=2) # [batch_size, passage_len]
        relevancy_degrees = tf.expand_dims(relevancy_degrees,axis=-1) # [batch_size, passage_len, 'x']
        in_passage_repres = tf.mul(in_passage_repres, relevancy_degrees)
        
    # =======Context Representation Layer & Multi-Perspective matching layer=====
    all_question_aware_representatins = []
    # max and mean pooling at word level
    all_question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keep_dims=True))
    all_question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keep_dims=True))
    question_aware_dim = 2
    
    if MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            max_att = cal_max_question_representation(in_question_repres, cosine_matrix)
            max_att_decomp_params = tf.get_variable("max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            max_attentive_rep = cal_attentive_matching(raw_in_passage_repres, max_att, max_att_decomp_params)
            all_question_aware_representatins.append(max_attentive_rep)
            question_aware_dim += MP_dim
    
    # lex decomposition
    if with_lex_decomposition:
        lex_decomposition = cal_linear_decomposition_representation(raw_in_passage_repres, passage_lengths, cosine_matrix,is_training, 
                                            lex_decompsition_dim, dropout_rate)
        all_question_aware_representatins.append(lex_decomposition)
        if lex_decompsition_dim== -1: question_aware_dim += 2 * input_dim
        else: question_aware_dim += 2* lex_decompsition_dim
        
    with tf.variable_scope('context_MP_matching'):
        for i in xrange(context_layer_num):
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

                    # question representation
                    (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                        sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                    in_question_repres = tf.concat(2, [question_context_representation_fw, question_context_representation_bw])

                    # passage representation
                    tf.get_variable_scope().reuse_variables()
                    (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                        sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                    in_passage_repres = tf.concat(2, [passage_context_representation_fw, passage_context_representation_bw])
                    
                # Multi-perspective matching
                with tf.variable_scope('MP_matching'):
                    (matching_vectors, matching_dim) = match_passage_with_question(passage_context_representation_fw, 
                                passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    all_question_aware_representatins.extend(matching_vectors)
                    question_aware_dim += matching_dim
        
    all_question_aware_representatins = tf.concat(2, all_question_aware_representatins) # [batch_size, passage_len, dim]

    if is_training:
        all_question_aware_representatins = tf.nn.dropout(all_question_aware_representatins, (1 - dropout_rate))
    else:
        all_question_aware_representatins = tf.mul(all_question_aware_representatins, (1 - dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("matching_highway"):
            all_question_aware_representatins = multi_highway_layer(all_question_aware_representatins, question_aware_dim,highway_layer_num)
        
    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    aggregation_input = all_question_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in xrange(aggregation_layer_num):
            with tf.variable_scope('layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_input, 
                        dtype=tf.float32, sequence_length=passage_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                aggregation_input = tf.concat(2, cur_aggregation_representation)# [batch_size, passage_len, 2*aggregation_lstm_dim]
        
    #
    aggregation_representation = tf.concat(1, aggregation_representation) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])

    return (aggregation_representation, aggregation_dim)
        
def bilateral_match_func1(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True):
    init_scale = 0.01
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    match_representation = []
    match_dim = 0
        
    reuse_match_params = None
    if with_left_match:
        reuse_match_params = True
        with tf.name_scope("match_passsage"):
            with tf.variable_scope("MP-Match", reuse=None, initializer=initializer):
                (passage_match_representation, passage_match_dim) = unidirectional_matching(in_question_repres, in_passage_repres,
                            question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                            with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                            with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                            with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                            with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                            with_attentive_match=with_attentive_match,
                            with_max_attentive_match=with_max_attentive_match)
                match_representation.append(passage_match_representation)
                match_dim += passage_match_dim
    if with_right_match:
        with tf.name_scope("match_question"):
            with tf.variable_scope("MP-Match", reuse=reuse_match_params, initializer=initializer):
                (question_match_representation, question_match_dim) = unidirectional_matching(in_passage_repres, in_question_repres, 
                            passage_lengths, question_lengths, mask, question_mask, MP_dim, input_dim, 
                            with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                            with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                            with_aggregation_highway, with_lex_decomposition,lex_decompsition_dim,
                            with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                            with_attentive_match=with_attentive_match,
                            with_max_attentive_match=with_max_attentive_match)
                match_representation.append(question_match_representation)
                match_dim += question_match_dim
    match_representation = tf.concat(1, match_representation)
    return (match_representation, match_dim)



def bilateral_match_func2(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True, with_mean_aggregation=True):

    cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres) # [batch_size, passage_len, question_len]
    cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
    cosine_matrix_transpose = tf.transpose(cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]

    # ====word level matching======
    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # max and mean pooling at word level
    question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keep_dims=True)) # [batch_size, passage_length, 1]
    question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keep_dims=True))# [batch_size, passage_length, 1]
    question_aware_dim += 2
    passage_aware_representatins.append(tf.reduce_max(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
    passage_aware_representatins.append(tf.reduce_mean(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
    passage_aware_dim += 2
    

    if MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            qa_max_att = cal_max_question_representation(in_question_repres, cosine_matrix)# [batch_size, passage_len, dim]
            qa_max_att_decomp_params = tf.get_variable("qa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            qa_max_attentive_rep = cal_attentive_matching(in_passage_repres, qa_max_att, qa_max_att_decomp_params)# [batch_size, passage_len, decompse_dim]
            question_aware_representatins.append(qa_max_attentive_rep)
            question_aware_dim += MP_dim

            pa_max_att = cal_max_question_representation(in_passage_repres, cosine_matrix_transpose)# [batch_size, question_len, dim]
            pa_max_att_decomp_params = tf.get_variable("pa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            pa_max_attentive_rep = cal_attentive_matching(in_question_repres, pa_max_att, pa_max_att_decomp_params)# [batch_size, question_len, decompse_dim]
            passage_aware_representatins.append(pa_max_attentive_rep)
            passage_aware_dim += MP_dim

    with tf.variable_scope('context_MP_matching'):
        for i in xrange(context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

                    # question representation
                    (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                        sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                    in_question_repres = tf.concat(2, [question_context_representation_fw, question_context_representation_bw])

                    # passage representation
                    tf.get_variable_scope().reuse_variables()
                    (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                        sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                    in_passage_repres = tf.concat(2, [passage_context_representation_fw, passage_context_representation_bw])
                    
                # Multi-perspective matching
                with tf.variable_scope('left_MP_matching'):
                    (matching_vectors, matching_dim) = match_passage_with_question(passage_context_representation_fw, 
                                passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    question_aware_representatins.extend(matching_vectors)
                    question_aware_dim += matching_dim
                
                with tf.variable_scope('right_MP_matching'):
                    (matching_vectors, matching_dim) = match_passage_with_question(question_context_representation_fw, 
                                question_context_representation_bw, question_mask,
                                passage_context_representation_fw, passage_context_representation_bw,mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    passage_aware_representatins.extend(matching_vectors)
                    passage_aware_dim += matching_dim
        

        
    question_aware_representatins = tf.concat(2, question_aware_representatins) # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(2, passage_aware_representatins) # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - dropout_rate))
    else:
        question_aware_representatins = tf.mul(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.mul(passage_aware_representatins, (1 - dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,highway_layer_num)
        
    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    
    '''
    if with_mean_aggregation:
        aggregation_representation.append(tf.reduce_mean(question_aware_representatins, axis=1))
        aggregation_dim += question_aware_dim
        aggregation_representation.append(tf.reduce_mean(passage_aware_representatins, axis=1))
        aggregation_dim += passage_aware_dim
    #'''

    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in xrange(aggregation_layer_num): # support multiple aggregation layer
            with tf.variable_scope('left_layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, qa_aggregation_input, 
                        dtype=tf.float32, sequence_length=passage_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                qa_aggregation_input = tf.concat(2, cur_aggregation_representation)# [batch_size, passage_len, 2*aggregation_lstm_dim]

            with tf.variable_scope('right_layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, pa_aggregation_input, 
                        dtype=tf.float32, sequence_length=question_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                pa_aggregation_input = tf.concat(2, cur_aggregation_representation)# [batch_size, passage_len, 2*aggregation_lstm_dim]
    #
    aggregation_representation = tf.concat(1, aggregation_representation) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
    
    return (aggregation_representation, aggregation_dim)

