'''
You are strongly recommended to run this code under keras 2.2.4
Several known issues would occur under keras 2.2.5 
'''

'''
BiLSTM structure reference:  https://spaces.ac.cn/archives/6810
Attention structure reference: https://spaces.ac.cn/archives/4765
Customrized keras layer reference https://spaces.ac.cn/archives/5765
'''



# -*- coding: utf-8 -*-

from tqdm import tqdm
import json
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd
import os
# from sklearn.utils import shuffle
import rouge as rouge


min_count = 5  #字出现最低频率,低于这个就去除
maxlen = 512   #序列最大长度
batch_size = 128  
epochs = 100
char_size = 256  # embedding size
z_dim = 128  # LSTM的hidden state的size

df = pd.read_csv('train.csv')
train_df = df
test_df = df.iloc[0:20]


'''
构建词典
chars
id2char
char2id
'''
if os.path.exists('vocab.json'):
    chars, id2char, char2id = json.load(open('vocab.json'))
    id2char = {int(i): j for i, j in id2char.items()}
else:
    chars = {}
    for text,summarization in tqdm(df.values):
        for w in text:  #数据集里的text
            chars[w] = chars.get(w, 0) + 1
        for w in summarization:  #数据集里的summarization
            chars[w] = chars.get(w, 0) + 1
    chars = {i: j for i, j in chars.items() if j >= min_count}
    # 0: padding
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i + 4: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([chars, id2char, char2id], open('vocab.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id,找不到的用1代替(即unk)
    if start_end:  # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen - 2]]
        ids = [2] + ids + [3]
    else:  # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


#============================================================


def padding(x):
    ml = max([len(i) for i in x])
    return [i + [0] * (ml - len(i)) for i in x]


def data_generator(data):
    # 数据生成器
    X, Y = [], []
    while True:
        for a in data.values:
            X.append(str2id(a[1]))
            Y.append(str2id(a[0], start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


def to_one_hot(x):
    x, x_mask = x
    x = K.cast(x, 'int32')
    # 得到 [0,0,0,1,0],[0,1,0,0,0]这样的向量,即句子里每个单词的one-hot表示
    # [batch_size, seq_len] --> [batch_size, seq_len, vocabulary_size+4], 单词数量+4(即len(char)+4)
    x = K.one_hot(x, len(chars) + 4) 
    # [batch_size, 1, vocabulary_size+4], 即变成每个句子里有多少个某个单词
    x = K.sum(x_mask * x, 1, keepdims=True) 
    # 大于1都是1, 小于1都是0
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x   # x的size是[batch_size, 1, vocabulary_size+4]


class ScaleShift(Layer):
    """
    缩放平移层（Scale and shift） Y=exp(b)*X+c
    """

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs, **kwargs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class OurLayer(Layer):
    """
    定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层。
    Keras 2.3之前Layer里直接套其他Layer是训练不了的， 所以需要这种方式来复用别的已经实现好的layer。2.3之后不需要了，可以直接layer套layer。
    """

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs


class OurBidirectional(OurLayer):
    """
    自己封装双向RNN，允许传入mask，保证对齐
    """
    '''
    为什么没有用Keras自带的Bidirectional()? 
    因为它只能把所有部分都反转过来,而不是仅仅把非pad部分反转.
    '''
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        mask_len = K.round(K.sum(mask, 1)[:, 0]) #统计mask有多长
        mask_len = K.cast(mask_len, 'int32')
        # 仅把有值的部分反转
        return K.tf.reverse_sequence(x, mask_len, seq_dim=1)

    def call(self, inputs, **kwargs):
        x, mask = inputs 
        # [batch_size, seq_len, hidden_size/2]
        x_forward = self.reuse(self.forward_layer, x)  #把原始数据[x_1,x_2,..x_k]送入正向层得到正向结果,结果是正序的y：[y_1,y_2,y_3...]
      
        # [batch_size, seq_len, hidden_size/2]
        x_backward = self.reverse_sequence(x, mask)     #把原始输入数据反过来(但是padding部分不参与反转,可以参照上面实现), 此时输入x是[x_k,x_k-1,...x_2,x_1]
        x_backward = self.reuse(self.backward_layer, x_backward) #反转的数据送入后向层得到反向结果,结果是逆序的y'：[y'_k,y'_k-1,...y'_2,y'_1]
        x_backward = self.reverse_sequence(x_backward, mask)  #再把y'反过来， 得到[y'_1,y'_2,...,y'_k-1]
        
        # 按最后一个维度拼起来. 
        # [batch_size, seq_len, hidden_size]
        x = K.concatenate([x_forward, x_backward], 2) 
        return x * mask

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)


class Attention(OurLayer):
    """
    multi-head+attention 多头注意力机制
    主要加入并修改了mask机制
    """

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads                   # default = 8
        self.size_per_head = size_per_head   # default = 16
        self.out_dim = heads * size_per_head # default = 128
        self.key_size = key_size if key_size else size_per_head # default = 16
        self.mask_right = mask_right

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':  
                return x * mask #这里把x做点乘操作,把不应该有值的位置变成0
            else:              
                return x - (1 - mask) * 1e10  #这里把0(即padding)的位置改成-inf, 为了后续softmax

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
                
        """
        If self-attention, then seq_len_q = seq_len_k = seq_len_v
        Else, seq_len_k = seq_len_v
        So I just use seq_len_kv here.
        """
        # [batch_size, seq_len_q, out_dim]
        qw = self.reuse(self.q_dense, q)
        # [batch_size, seq_len_kv, out_dim]
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        
        # [batch_size, seq_len_q, n_head, head_size]
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        # [batch_size, seq_len_kv, n_head, head_size]
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))

        # [batch_size, n_head, seq_len_q, head_size]
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        # [batch_size, n_head, seq_len_kv, head_size]
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))

        """
        Try to use tf.matmul if K.batch_dot doesn't work
        e.g.
        a = tf.matmul(qw, K.permute_dimensions(kw, (0,1,3,2)))
        Keypoint is: get correct shape
        """
        # [batch_size, n_head, seq_len_q, seq_len_kv]
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5 
        # [batch_size, seq_len_kv, seq_len_q, n_head]
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        # v_mask shape = [batch_size, seq_len_kv, 1]
        a = self.mask(a, v_mask, 'add')  
        # [batch_size, n_head, seq_len_q, seq_len_kv]
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)  # softmax on seq_len_kv
        

        # [batch_size, n_head, seq_len_q, head_size] 
        o = K.batch_dot(a, vw, [3, 2])  
        # [batch_size, seq_len_q, n_head, head_size]
        o = K.permute_dimensions(o, (0, 2, 1, 3))   
        # [batch_size, seq_len_q, out_dim]
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))  
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)



#  主要模型入口
def graph():
    x_in = Input(shape=(None,)) # [batch_size, seq_len_x]
    y_in = Input(shape=(None,)) # [batch_size, seq_len_y]
    x, y = x_in, y_in

    
    # [batch_size, seq_len, 1]  
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x) # 就是[1,1,1,1,0,0],[1,1,0,0,0,0],....
    y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

    #为了引入先验知识, 把整篇文章做one hot处理
    # [batch_size, 1, vocabulary_size+4]
    x_one_hot = Lambda(to_one_hot)([x, x_mask])
    x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（标题的字词很可能在文章出现过）

    # vocabulary_size+4 --> embedding_size
    embedding = Embedding(len(chars) + 4, char_size)
    # [batch_size, seq_len, embedding_size]
    x = embedding(x) #就是text
    y = embedding(y) #就是summarization

    # encoder，双层双向LSTM
    # [batch_size, seq_len, hidden_size]
    x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    #如果是CPU请换成这行
    #x = OurBidirectional(LSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    x = LayerNormalization()(x)
    x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    #如果是CPU请换成这行
    #x = OurBidirectional(LSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    x = LayerNormalization()(x)


    # decoder，双层单向LSTM
    # [batch_size, seq_len, hidden_size]
    y = CuDNNLSTM(z_dim, return_sequences=True)(y)
    #如果是CPU请换成这行
    #y = LSTM(z_dim, return_sequences=True)(y)
    y = LayerNormalization()(y)
    y = CuDNNLSTM(z_dim, return_sequences=True)(y)
    #如果是CPU请换成这行
    #y = LSTM(z_dim, return_sequences=True)(y)
    y = LayerNormalization()(y)

    # [batch_size, seq_len_y, attention_out_dim]
    xy = Attention(8, 16)([y, x, x, x_mask])
    # [batch_size, seq_len_y, hidden_size+attention_out_dim]
    xy = Concatenate()([y, xy])

    # 输出分类
    # [batch_size, seq_len_y, embedding_size]
    xy = Dense(char_size)(xy)
    xy = Activation('relu')(xy)
    # [batch_size, seq_len_y, vocabulary_size + 4]
    xy = Dense(len(chars) + 4)(xy) 
    # [batch_size, seq_len_y, vocabulary_size + 4]
    xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
    xy = Activation('softmax')(xy)  # do softmax on vocabulary_size

    # 多分类交叉熵作为loss，但mask掉padding部分
    cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
    cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

    model = Model([x_in, y_in], xy)
    # model.load_weights('best_model.weights')
    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(1e-3))
    return model


def gen_sent(s, model, topk=3, maxlen=64):
    """
    beam search解码, 有一个参数k,即选择k个最优结果.
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    # 输入转id
    xid = np.array([str2id(s)] * topk)
    # 解码均以<start>开头，这里<start>的id为2
    yid = np.array([[2]] * topk)  #[[2],[2],[2]]
    # 候选答案分数
    scores = [0] * topk
    
    #两种停止情况, 碰到<end>或者最长maxlen
    for i in range(maxlen):
        # 直接忽略<padding>、<unk>、<start> 因为解码时不可能出现这些
        proba = model.predict([xid, yid])[:, i, 3:]
        # 取对数，方便计算
        log_proba = np.log(proba + 1e-6)
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]
        
        _yid = []
        _scores = []
        
        # 假如是第一次,那就只需要计算topk个值
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j] + 3]) #这边要+3, 因为前面去掉了<padding>、<unk>、<start>
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
                
        # 非第一词需要计算topk^2个值
        else:
            # 遍历topk*topk的组合
            for j in range(topk):
                for k in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            # 从中选出新的topk
            _arg_topk = np.argsort(_scores)[-topk:]
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)
        
        # 检查是否存在<end>
        ends = np.where(yid[:, -1] == 3)[0]
        if len(ends) > 0:
            k = ends[scores[ends].argmax()]
            return id2str(yid[k])
    
    return id2str(yid[np.argmax(scores)])


s1 = '四海网讯，近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人<Paragraph>接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。2015年3月21日，章子怡担任制片人的电影《从天儿降》开机，在开机发布会上几张合影，让网友又燃起了好奇心：“章子怡真的怀孕了吗?”但后据证实，章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日，《太平轮》新一轮宣传，章子怡又被发现状态不佳，不时深呼吸，不自觉想捂住肚子，又觉得不妥。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期，相信9月26日的演唱会应该还会有惊喜大白天下吧。'
s2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'



def print_sentence(s,model):
    # 查看某个具体句子实际效果
    print(gen_sent(s, self.model))



class Evaluate(Callback):
    def __init__(self, model):
        self.lowest = 1e10
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子
        print_sentence(s1, self.model)
        print_sentence(s2, self.model)

        summarization = test_df['summarization'].values
        text = test_df['text'].values
        pred = []
        for t, s in tqdm(zip(text, summarization)):
            pred.append(gen_sent(t, self.model))

        rouge_1 = rouge.Rouge().get_scores(pred, summarization.tolist())[0]['rouge-1']['f']
        rouge_2 = rouge.Rouge().get_scores(pred, summarization.tolist())[0]['rouge-2']['f']
        rouge_l = rouge.Rouge().get_scores(pred, summarization.tolist())[0]['rouge-l']['f']
        print('rouge-1:', rouge_1)
        print('rouge-2:', rouge_2)
        print('rouge-l:', rouge_l)

        # 保存最优模型
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('best_model.weights')


if __name__ == '__main__':
    model = graph()
    evaluator = Evaluate(model)
    steps = len(df) / batch_size
    # 因为数据量不小,使用了generator慢慢的产生数据, 所以这里用fit_generator而不是fit
    model.fit_generator(data_generator(train_df),
                        steps_per_epoch=steps,
                        epochs=epochs,
                        callbacks=[evaluator])
