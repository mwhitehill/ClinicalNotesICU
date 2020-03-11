'''
In intensive care units, where patients come in with a wide range of health conditions, 
triaging relies heavily on clinical judgment. ICU staff run numerous physiological tests, 
such as bloodwork and checking vital signs, 
to determine if patients are at immediate risk of dying if not treated aggressively.
'''

import tensorflow as tf
from tensorflow.contrib import rnn
import pickle, random, socket, sys, os, time
import numpy as np
from datetime import datetime
sys.path.append(os.getcwd())

import utils
from config import Config
from mimic3models import common_utils
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.in_hospital_mortality import utils as ihm_utils
from models.tacotron_models import TacotronEncoderCell, EncoderConvolutions, EncoderRNN

alpha = .9

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info("*** Loaded Data ***")

conf = utils.get_config()
args = utils.get_args()
# [datetime.now().strftime('%Y.%m.%d_%H-%M-%S'),
folder_name = '_'.join(["ihm", args['name']])

for i in range(100):
    folder_name_n = folder_name + '_' + str(i)
    log_folder = os.path.join(conf.log_folder, folder_name_n)
    if not(os.path.exists(log_folder)):
        break

os.makedirs(log_folder)

log = utils.get_logger(log_folder, args['log_file'])
results_csv_path = os.path.join(log_folder, 'results.csv')
header = "epoch;loss_ts;loss_txt;loss;AUCPR;AUCROC;val_loss_ts;val_loss_txt;val_loss;val_AUCPR;val_AUCROC\n"
with open(results_csv_path, 'w') as handle:
    handle.write(header)

model_name = args['model_name']
assert model_name in ['baseline', 'avg_we', 'transformer', 'cnn', 'text_only','cnn_gru','gru','tacotron']

tf.logging.info("MODEL_NAME: {}".format(model_name))

vectors, word2index_lookup = utils.get_embedding_dict(conf, args['TEST'])
lookup = utils.lookup
# let's set pad token to zero padding instead of random padding.
# might be better for attention as it will give minimum value.
if conf.padding_type == 'Zero':
    tf.logging.info("Zero Padding..")
    vectors[lookup(word2index_lookup, '<pad>')] = 0

tf.logging.info(str(vars(conf)))
tf.logging.info(str(args))

number_epoch = int(args['number_epoch'])
batch_size = int(args['batch_size'])

X = tf.placeholder(shape=(None, 48, 76), dtype=tf.float32, name='X')  # B*48*76
y = tf.placeholder(shape=(None), dtype=tf.float32, name='y')
text = tf.placeholder(shape=(None, None), dtype=tf.int32, name='text')  # B*L
text_lens = tf.placeholder(shape=(None), dtype=tf.int32, name='text_lens')
dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_kp')
T = tf.placeholder(dtype=tf.int32)
N = tf.placeholder(dtype=tf.int32)

# Have a big enough gpu and simple enough model to fit word embeddings on gpu (all other cases, can't do that)
# if socket.gethostname() == 'area51.cs.washington.edu' and model_subname == 'none':
#     W_place = tf.placeholder(shape=vectors.shape, dtype = tf.float32, name='W_place')
#     W = tf.Variable(W_place, name="W", trainable=False, shape = vectors.shape)
# else:
with tf.device('/CPU:0'):
    W_place = tf.placeholder(shape=vectors.shape, dtype = tf.float32, name='W_place')
    W_place_concat = tf.concat([W_place,tf.constant(0, dtype=tf.float32, shape=(1,vectors.shape[1]))],axis=0)
    W = tf.Variable(W_place_concat, name="W", trainable=False, shape = (vectors.shape[0]+1, vectors.shape[1]))

embeds = tf.nn.embedding_lookup(W, text)

hidden_units = vectors.shape[1]

#avg_word_embedding_mode = bool(int(args['avg_we_model']))
#baseline = bool(int(args['baseline']))

if model_name != 'baseline':
    if model_name == 'avg_we':
        tf.logging.info("Average Word Embeddings Model!")
        text_embeddings = tf.math.reduce_mean(embeds, axis=1, keepdims=False)
    elif model_name == 'transformer':
        tf.logging.info("Transformer Encoder Based Model!")
        key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(embeds), axis=-1)), -1)
        embeds += utils.positional_encoding(text, T, N, num_units=hidden_units,zero_pad=False, scale=False, scope="enc_pe")
        embeds *= key_masks

        # Dropout
        embeds = tf.nn.dropout(embeds, keep_prob=dropout_keep_prob)
        enc = embeds
        # Blocks
        for i in range(conf.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                enc = utils.multihead_attention(queries=enc,
                                                keys=embeds,
                                                num_units=hidden_units,
                                                num_heads=10,
                                                dropout_rate=dropout_keep_prob,
                                                causality=False)

                # Feed Forward
                enc = utils.feedforward(enc, num_units=[4*hidden_units, hidden_units])
        text_embeddings = tf.reduce_mean(enc, axis=1)
    elif model_name == 'gru':
        tf.logging.info("GRU Model")
        rnn_cell_text = rnn.GRUCell(num_units=200, name='gru_text')
        rnn_outputs_text, rnn_outputs_text_last = tf.nn.dynamic_rnn(rnn_cell_text, embeds, dtype=tf.float32, sequence_length=text_lens)
        rnn_outputs_text_last = tf.nn.dropout(rnn_outputs_text_last, keep_prob=dropout_keep_prob)
    elif model_name == 'tacotron':
        rnn_cell_text = TacotronEncoderCell(EncoderConvolutions(True, scope='encoder_convolutions'),
                                           EncoderRNN(True, size=256,zoneout=.1, scope='encoder_LSTM'))
        rnn_outputs_text_last = rnn_cell_text(embeds, text_lens)
        rnn_outputs_text_last = tf.nn.dropout(rnn_outputs_text_last, keep_prob=.8)
    elif model_name == 'cnn_gru':
        start = 2
        end = 4
        sizes = range(start, end)
        pool_size = 200
        filters = 16
        rnn_nodes = 8
        GRU = False
        dropout = dropout_keep_prob
        result_tensors = []

        for ngram_size in sizes:
            # 256 -> 2,3 best yet.
            text_conv1d = tf.layers.conv1d(inputs=embeds, filters=filters, kernel_size=ngram_size,
                                           strides=1, padding='same', dilation_rate=1,
                                           activation='relu', name='Text_Conv_1D_N{}'.format(ngram_size),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            # text_conv1d = tf.reduce_max(text_conv1d, axis=1, keepdims=False)
            result_tensors.append(text_conv1d)
        text_embeddings = tf.concat(result_tensors, axis=-1)
        text_embeddings_pooled = tf.nn.max_pool1d(input=text_embeddings, ksize=pool_size, strides=pool_size,
                                                  padding='SAME', name='Text_Max_Pool_1D')

        text_lens_div = tf.cast(tf.math.floor(tf.math.divide(text_lens,pool_size)),dtype=tf.int32)

        if GRU:
            rnn_cell_text = rnn.GRUCell(num_units=rnn_nodes, name='gru_text')
            rnn_outputs_text, rnn_outputs_text_last = tf.nn.dynamic_rnn(rnn_cell_text, text_embeddings_pooled, dtype=tf.float32,
                                                                    sequence_length=text_lens_div)
        else:
            rnn_cell_text = rnn.LSTMCell(num_units=rnn_nodes, name='lstm_text')
            _, rnn_outputs_text_last = tf.nn.dynamic_rnn(rnn_cell_text, text_embeddings_pooled, dtype=tf.float32,
                                                sequence_length=text_lens_div)
            rnn_outputs_text_last= rnn_outputs_text_last.h

        rnn_outputs_text_last = tf.nn.dropout(rnn_outputs_text_last, keep_prob=dropout)#dropout_keep_prob)
        # else:
        #     rnn_cell_text_fw = rnn.LSTMCell(num_units=256, name='lstm_text_fw')
        #     rnn_cell_text_bw = rnn.LSTMCell(num_units=256, name='lstm_text_bw')
        #     _, rnn_outputs_text_last = tf.nn.bidirectional_dynamic_rnn(rnn_cell_text_fw, rnn_cell_text_bw, text_embeddings,
        #                                                                time_major=False, dtype=tf.float32,sequence_length=text_lens)
        #     rnn_outputs_text = tf.concat([rnn_outputs_text_last[0].h, rnn_outputs_text_last[1].h],1)#, #2)
        #     rnn_outputs_text_last = tf.layers.dense(inputs=rnn_outputs_text_last, units=512, activation='relu',
        #                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        #     rnn_outputs_text_last = tf.nn.dropout(rnn_outputs_text_last, keep_prob=dropout_keep_prob)


    else:
        tf.logging.info("1D Convolution Model")
        start =2
        end = 4
        sizes = range(start,end)#5)
        filters = 256
        filters_2d = 4
        pool_size = 250
        result_tensors = []
        result_tensors_am = []
        pool=False

        for ngram_size in sizes:
            # 256 -> 2,3 best yet.
            text_conv1d = tf.layers.conv1d(inputs=embeds, filters=filters, kernel_size=ngram_size,
                                           strides=1, padding='same', dilation_rate=1,
                                           activation='relu', name='Text_Conv_1D_N{}'.format(ngram_size),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            if pool:
                text_conv1d = tf.nn.max_pool1d(input=text_conv1d, ksize=pool_size, strides=pool_size,
                                                      padding='SAME', name='Text_Max_Pool_1D')
            else:
                text_conv1d = tf.reduce_max(text_conv1d, axis=1, keepdims=False)
            result_tensors.append(text_conv1d)

        text_embeddings = tf.concat(result_tensors, axis=-1) #axis=1)

        if pool:
            # text_embeddings = tf.reshape(text_embeddings, [-1, int(conf.max_len / pool_size) * filters * (end-start)])

            nodes = int(conf.max_len / pool_size)
            text_embeddings = tf.layers.conv2d(inputs=tf.expand_dims(text_embeddings,-1), filters=filters_2d, kernel_size = (nodes,1), strides=(nodes,1), padding='same',
                                  activation='relu', name='Text_Conv_2D',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            text_embeddings = tf.reshape(text_embeddings, [-1, filters * (end-start) * filters_2d])

        text_embeddings = tf.nn.dropout(text_embeddings, keep_prob=dropout_keep_prob)

rnn_cell = rnn.LSTMCell(num_units=256, name='lstm_main')
rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, X, time_major=False, dtype=tf.float32)

#mean_rnn_outputs = tf.math.reduce_mean(rnn_outputs, axis=1, keepdims=False)
mean_rnn_outputs = rnn_outputs[:, -1, :]
if model_name == 'baseline':
    logit_X = mean_rnn_outputs
elif model_name == 'text_only':
    logit_X = text_embeddings
elif model_name in ['cnn_gru','gru','tacotron']:
    if args['split_loss']:
        logit_X = mean_rnn_outputs
        logit_text = rnn_outputs_text_last
    else:
        logit_X = tf.concat([rnn_outputs_text_last, mean_rnn_outputs], axis=1)
else:
    logit_X = tf.concat([text_embeddings, mean_rnn_outputs], axis=1)

logits_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
logits = tf.layers.dense(inputs=logit_X, units=1, activation=None, use_bias=False,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         kernel_regularizer=logits_regularizer)
logits = tf.squeeze(logits)
loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits), name='celoss')
probs = tf.math.sigmoid(logits)

if args['split_loss']:
    logits_t = tf.layers.dense(inputs=logit_text, units=1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=logits_regularizer)
    logits_t = tf.squeeze(logits_t)
    loss_t = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits_t), name='celoss_t')
    probs_t = tf.math.sigmoid(logits_t)

    loss_full = loss * alpha + loss_t * (1-alpha)
    probs_full = probs * alpha + probs_t * (1-alpha)
else:
    loss_t=tf.constant(0)
    loss_full = loss
    probs_full = probs

loss_full += tf.losses.get_regularization_loss()

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
    train_op = optimizer.minimize(
        loss=loss_full,
        global_step=tf.train.get_global_step())

with tf.name_scope('train_metric'):
    aucroc, update_aucroc_op = tf.metrics.auc(labels=y, predictions=probs)
    aucpr, update_aucpr_op = tf.metrics.auc(labels=y, predictions=probs,
                                            curve="PR")

with tf.name_scope('valid_metric'):
    val_aucroc, update_val_aucroc_op = tf.metrics.auc(
        labels=y, predictions=probs)
    val_aucpr, update_val_aucpr_op = tf.metrics.auc(labels=y, predictions=probs,
                                                    curve="PR")

loss_summary = tf.summary.scalar(name='loss', tensor=loss_full)
aucroc_summary = tf.summary.scalar(name='aucroc', tensor=aucroc)
aucpr_summary = tf.summary.scalar(name='aucpr', tensor=aucpr)
summ_tr = tf.summary.merge([loss_summary, aucroc_summary, aucpr_summary])

# loss_summary_val = tf.placeholder(tf.float32, shape=(), name="loss_summary_val")
# summ_val_loss = tf.summary.scalar("loss_val", loss_summary_val)
aucroc_summary_val = tf.summary.scalar(name='aucroc_val', tensor=val_aucroc)
aucpr_summary_val = tf.summary.scalar(name='aucpr_val', tensor=val_aucpr)

if not(args['TEST_MODEL']):
    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(conf.ihm_path, 'train'),
                                             listfile=os.path.join(conf.ihm_path, 'train_listfile.csv'), period_length=48.0)

    discretizer = Discretizer(timestep=float(conf.timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    # choose here which columns to standardize
    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = conf.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(conf.timestep, conf.imputation)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    normalizer = None
    train_raw = ihm_utils.load_data(train_reader, discretizer, normalizer, conf.small_part, return_names=True)

    print("Number of train_raw_names: ", len(train_raw['names']))

    text_reader = utils.TextReader(conf.textdata_fixed, conf.starttime_path)

    train_text = text_reader.read_all_text_concat_json(train_raw['names'], 48)
    data = utils.merge_text_raw(train_text, train_raw)

    data_X = data[0]
    data_y = data[1]
    data_text = data[2]
    del data
    del train_raw
    del train_text

    eval_reader = InHospitalMortalityReader(dataset_dir=os.path.join(conf.ihm_path, 'train'),listfile=os.path.join(conf.ihm_path, 'val_listfile.csv'),
                                            period_length=48.0)
    eval_raw = ihm_utils.load_data(eval_reader, discretizer,normalizer, conf.small_part, return_names=True)

    eval_text = text_reader.read_all_text_concat_json(eval_raw['names'], 48)
    data = utils.merge_text_raw(eval_text, eval_raw)

    eval_data_X = data[0]
    eval_data_y = data[1]
    eval_data_text = data[2]
    del data
    del eval_raw
    del eval_text

if args['mode'] == 'test':
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(conf.ihm_path, 'test'),
                                            listfile=os.path.join(
                                                conf.ihm_path, 'test_listfile.csv'),
                                            period_length=48.0)
    test_raw = ihm_utils.load_data(test_reader, discretizer,
                                   normalizer, conf.small_part, return_names=True)
    text_reader_test = utils.TextReader(
        conf.test_textdata_fixed, conf.test_starttime_path)
    test_text = text_reader_test.read_all_text_concat_json(
        test_raw['names'], 48)
    data = utils.merge_text_raw(test_text, test_raw)

    test_data_X = data[0]
    test_data_y = data[1]
    test_data_text = data[2]

    del data
    del test_raw
    del test_text

lens_all = []

def generate_tensor_text(t, w2i_lookup):
    global lens_all
    t_new = []
    max_len = -1

    lens_all += [len(text) for text in t]

    for text in t:
        tokens = list(map(lambda x: lookup(w2i_lookup, x), str(text).split()))
        if conf.max_len > 0:
            tokens = tokens[:conf.max_len] if args['no_flip_text_start'] else tokens[-conf.max_len:]
        t_new.append(tokens)
        max_len = max(max_len, len(tokens))
    pad_token = w2i_lookup['<pad>']
    lens = [len(t_n) for t_n in t_new]
    for i in range(len(t_new)):
        if len(t_new[i]) < max_len:
            if args['no_flip_text_start']:
                t_new[i] += [pad_token] * (max_len - len(t_new[i]))
            else:
                t_new[i]  = [pad_token] * (max_len - len(t_new[i])) + t_new[i]
    return (np.array(t_new), np.array(lens))

def generate_padded_batches(x, y, t, bs, w2i_lookup):
    batches = []
    begin = 0
    while begin < len(t):
        end = min(begin+batch_size, len(t))
        x_slice = np.stack(x[begin:end])
        y_slice = np.stack(y[begin:end])
        t_slice, lens = generate_tensor_text(t[begin:end], w2i_lookup)
        batches.append((x_slice, y_slice, t_slice, lens))
        begin += batch_size
        if args['TEST']:
            break
    # print("MAX LEN", max(lens_all), np.median(lens_all), np.mean(lens_all))
    return batches


def validate(data_X_val, data_y_val, data_text_val, batch_size, word2index_lookup,
             sess, saver, last_best_val_aucpr, loss, val_aucpr, val_aucroc,
             update_val_aucpr_op, update_val_aucroc_op, save, epoch=0, summ_writer=None):
    val_batches = generate_padded_batches(data_X_val, data_y_val, data_text_val, batch_size, word2index_lookup)
    loss_list = []
    loss_list_ts = []
    loss_list_txt = []
    aucpr_obj = utils.AUCPR()
    #sess.run(tf.variables_initializer([v for v in tf.local_variables() if 'valid_metric' in v.name]))
    sess.run(tf.local_variables_initializer(), {W_place: vectors})
    for v_batch in val_batches:
        fd = {X: v_batch[0], y: v_batch[1],
              text: v_batch[2], text_lens: v_batch[3], dropout_keep_prob: 1,
              T: v_batch[2].shape[1],
              N: v_batch[2].shape[0]}
        loss_value, loss_value_ts, loss_value_txt,_, _, probablities = sess.run(
            [loss_full, loss, loss_t, update_val_aucpr_op, update_val_aucroc_op, probs], fd)
        loss_list.append(loss_value)
        loss_list_ts.append(loss_value_ts)
        loss_list_txt.append(loss_value_txt)
        aucpr_obj.add(probablities, v_batch[1])

    # summ_loss_val_res = sess.run(summ_val_loss, {loss_summary_val: np.mean(loss_list)})
    final_aucpr, summ_aucpr_val = sess.run([val_aucpr, aucpr_summary_val])
    final_aucroc, summ_aucroc_val = sess.run([val_aucroc, aucroc_summary_val])

    if summ_writer is not None:
        # summ_writer.add_summary(summ_loss_val_res, epoch)
        summ_writer.add_summary(summ_aucpr_val, epoch)
        summ_writer.add_summary(summ_aucroc_val, epoch)
        with open(results_csv_path, 'a') as handle:
            handle.write("{};{};{};{};{}\n".format(np.mean(loss_list_ts),np.mean(loss_list_txt),np.mean(loss_list), final_aucpr, final_aucroc))

    tf.logging.info("Val Loss TS: %f - Val Loss Txt: %f - Validation Loss: %f - AUCPR: %f - AUCPR-SKLEARN: %f - AUCROC: %f" %
                    (np.mean(loss_list_ts), np.mean(loss_list_txt), np.mean(loss_list), final_aucpr, aucpr_obj.get(), final_aucroc))

    # aucpr_obj.save()
    changed = True if final_aucpr > last_best_val_aucpr else False
    if (epoch + 1) % 20 == 0 and save:
        save_path = saver.save(sess, os.path.join(log_folder, 'ckpt_e{}'.format(epoch)))
        tf.logging.info("Model saved in path: %s" % save_path)
    return max(last_best_val_aucpr, final_aucpr), changed

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
print(len(tf.local_variables()))
saver = tf.train.Saver(max_to_keep=1)
last_best_val_aucpr = -1


gpu_config = tf.ConfigProto(device_count={'GPU': 1})
#gpu_config.allow_soft_placement=True

FIRST = True
with tf.Session(config=gpu_config) as sess:

    summ_writer = tf.summary.FileWriter(log_folder, sess.graph)

    sess.run(init, {W_place: vectors})

    if args['TEST_MODEL']:

        with open('example_batch.pkl', 'rb') as handle:
            batch = pickle.load( handle)

        fd = {X: batch[0], y: batch[1],
              text: batch[2], text_lens: batch[3], dropout_keep_prob: conf.dropout}

        # t1, t2, t3,t4,t5 = sess.run([text_embeddings, text_lens, text_lens_div,text_embeddings_pooled,rnn_outputs_text_last ],fd) #loss, probs rnn_outputs_text_last
        # t1, t2 = sess.run([loss, probs],fd)
        t1,t2= sess.run([text_conv1d, text_embeddings], fd) #loss, probs rnn_outputs_text_last, text_embeddings

        print(t1.shape)
        print(t2.shape)
        raise


    if bool(int(args['load_model'])):
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(conf.log_folder, args['checkpoint_path'])))
        # last_best_val_aucpr, _ = validate(eval_data_X, eval_data_y, eval_data_text,
        #                                   batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
        #                                   loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, False)

    if args['mode'] == 'eval':
        assert bool(int(args['load_model']))
        tf.logging.info('Just Evaluating Mode.')
        last_best_val_aucpr, _ = validate(eval_data_X, eval_data_y, eval_data_text,
                                          batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
                                          loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, False)
        sys.exit(0)

    if args['mode'] == 'test':
        assert bool(int(args['load_model']))
        tf.logging.info('Testing Mode.')
        last_best_val_aucpr, _ = validate(test_data_X, test_data_y, test_data_text,
                                          batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
                                          loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, False)
        sys.exit(0)


    early_stopping = 0
    times = np.array([])
    for epoch in range(number_epoch):
        tf.logging.info("Started training for epoch: %d" % epoch)
        data = list(zip(data_X, data_y, data_text))
        random.shuffle(data)
        data_X, data_y, data_text = zip(*data)

        loss_list = []
        loss_list_ts = []
        loss_list_txt = []

        del data

        batches = generate_padded_batches(data_X, data_y, data_text, batch_size, word2index_lookup)

        tf.logging.info("Generated batches for the epoch!")


        for ind, batch in enumerate(batches):
            # train the batch and update parameters.

            if FIRST:
                start = time.time()

            fd = {X: batch[0], y: batch[1],
                  text: batch[2], text_lens: batch[3], dropout_keep_prob: conf.dropout,
                  T: batch[2].shape[1],
                  N: batch[2].shape[0]}

            _, loss_value,loss_value_ts,loss_value_txt, aucpr_value, aucroc_value, summ = sess.run(
                [train_op, loss_full, loss, loss_t, update_aucpr_op, update_aucroc_op, summ_tr], fd)
            loss_list.append(loss_value)
            loss_list_ts.append(loss_value_ts)
            loss_list_txt.append(loss_value_txt)

            if FIRST:
                end = time.time()
                times = np.append(times, end-start)
            if (ind + 1) % 250 == 0:
                print("Finished:", ind + 1, "of", len(batches))
                if FIRST:
                    print("Average time per batch (s): {:.4f}".format(np.mean(times)))
                    times = np.array([])
                    FIRST = False

        summ_writer.add_summary(summ, epoch)


            # if ind % 200 == 0:
            # tf.logging.info(
            #   "Processed Batch: %d for Epoch: %d" % (ind, epoch))

        current_aucroc = sess.run(aucroc)
        current_aucpr = sess.run(aucpr)

        with open(results_csv_path, 'a') as handle:
            handle.write("{};{};{};{};{};{};".format(epoch, np.mean(loss_list_ts),np.mean(loss_list_txt), np.mean(loss_list), current_aucpr, current_aucroc))


        tf.logging.info("Loss_ts: %f - Loss_txt: %f - Loss: %f - AUCPR: %f - AUCROC: %f" %
                        (np.mean(loss_list_ts), np.mean(loss_list_txt), np.mean(loss_list), current_aucpr, current_aucroc))

        # reset aucroc and aucpr local variables
        # sess.run(tf.variables_initializer(
        #    [v for v in tf.local_variables() if 'train_metric' in v.name]))
        sess.run(tf.local_variables_initializer(),{W_place: vectors})
        loss_list = []

        del batches
        tf.logging.info("Started Evaluation After Epoch : %d" % epoch)
        last_best_val_aucpr, changed = validate(eval_data_X, eval_data_y, eval_data_text,
                                                batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
                                                loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, True,
                                                epoch=epoch, summ_writer=summ_writer)
        if changed == False:
            early_stopping += 1
            tf.logging.info("Didn't improve!: " + str(early_stopping))
        else:
            early_stopping = 0

        if early_stopping >= 15:
            tf.logging.info(
                "AUCPR didn't change from last 15 epochs, early stopping")
            break
        tf.logging.info("*End of Epoch.*\n")
