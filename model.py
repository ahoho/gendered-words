import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import pickle
from collections import defaultdict

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

import pandas as pd
import numpy as np
import tensorflow as tf

'''
Data processing
'''

def save_object(obj, fpath):
    """
    Pickle an object and save it to file
    """
    with open(fpath, 'wb') as o:
        pickle.dump(obj, o)

def load_object(fpath):
    """
    Load a pickled object from file
    """
    with open(fpath, 'rb') as i:
        return pickle.load(i)


def parse_noun_map(fpath):
    '''
    Parse the noun-attribute map from file
    '''
    with open(fpath, 'r') as o:
        lines = list(o)
    noun_map = []
    for line in lines[1:]:
        msc_sg, msc_pl, fem_sg, fem_pl, neu_sg, neu_pl = line.strip().split(',')
        noun_map += [
            (msc_sg, msc_sg, 'msc', 'sg',),
            (msc_pl, msc_sg, 'msc', 'pl',),
            (fem_sg, msc_sg, 'fem', 'sg',),
            (fem_pl, msc_sg, 'fem', 'pl',),
        ]
    return pd.DataFrame(noun_map, columns=('noun', 'lemma', 'fem', 'plural'))


def merge_clean_diachronic_data(fpaths, noun_map, pos=None):
    '''
    Clean and merge the diachronic data

    Args
        fpath : list or str
            Path to data files
        cols : list fo str
            Column names to read in
        noun_map : dataframe
            Map nouns to attributes
        pos : str, one of wn.VERB or wn.ADJ
            Specify POS for cleaning
    '''
    if isinstance(fpaths, str):
        fpaths = list(fpaths)
    if pos is None:
        pos = wn.ADJ if all('jj' in f for f in fpaths) else wn.VERB
    
    data = pd.concat([
        pd.read_csv(fpath, sep=r'\s+', names=['w', 'noun', 'count'], na_filter=False)
        for fpath in fpaths
    ], ignore_index=True)
    
    # ensure that nouns and words are in correct columns
    not_nouns = ~data.noun.isin(noun_map.noun)
    data.loc[not_nouns, 'noun'], data.loc[not_nouns, 'w'] = (
        data.loc[not_nouns, 'w'], 
        data.loc[not_nouns, 'noun']
    )

    # clean up part of speech
    correct_pos = [len(wn.synsets(word, pos=pos)) > 0 for word in data.w]
    data = data.loc[correct_pos]

    # lemmatize if a verb
    if pos == wn.VERB:
        wn_lemmatizer = WordNetLemmatizer()
        data['w'] = [wn_lemmatizer.lemmatize(word, pos=wn.VERB) for word in data.w]

    # group the counts together
    data.groupby(['w', 'noun'], as_index=False, sort=False).sum()
    return data


def parse_collapsed_data(fpath, cols, noun_map=None, clean=False, pos=None, min_count=0):
    '''
    Parses the `(adjective/verb noun, lemma, gender, plural, count)` data
    from a file and builds a vocabulary

    Args
        fpath : str
            Path to data
        cols : list fo str
            Column names to read in
        noun_map : dataframe
            Map nouns to attributes
        min_count : int
            Minimum count of (adjective/verb noun) pairs allowed
    Returns
        data : dataframe
            A dataframe of the collapsed, parsed data
        w_vocab : dict
            A dictionary mapping words to tokens
    '''
    data = pd.read_csv(
        fpath,
        sep=r'\s+',
        names=cols,
    )
    
    # filter out counts too low
    w_count = data.groupby('w')['w'].count()
    data = data.loc[data.w.isin(w_count[w_count >= min_count].index)]

    if clean:
        merge_clean_diachronic_data([fpath], noun_map, pos)

    # map noun attributes if not in data
    if noun_map is not None:
        data = data.merge(noun_map, how='left', on='noun')
    
    # generate adjective/verbvocabulary and overwrite columns
    w_vocab = defaultdict(lambda: len(w_vocab))
    data['w'] = [w_vocab[a] for a in data.w]
    w_vocab = dict(w_vocab) # freeze dictionary

    # also create lemma vocab
    lemma_vocab = defaultdict(lambda: len(lemma_vocab))
    data['lemma'] = [lemma_vocab[n] for n in data.lemma]
    lemma_vocab = dict(lemma_vocab)

    # map to binary
    data['fem'] = (data['fem'] == 'fem') * 1
    data['plural'] = (data['plural'] == 'pl') * 1

    return data[['w', 'noun', 'lemma', 'fem', 'plural', 'count']], w_vocab, lemma_vocab


def parse_sentiment_data(fpath, vocab):
    '''
    Map adjectives/verbs to sentiments
    Args
        fpath : str
            Path to the sentiment data
        vocab : dict
            Vocabulary mapping strings to integer ids
        sent_lambda : float
            Scale parameter used when generating alpha
    Returns
        alpha : (|V|, 3) numpy array of floats
            alpha that parametrizes P(S | W) ~ Dir(alpha)
    '''
    # map sentiments and handle missing sentiments
    sents = pd.read_csv(fpath, index_col=0)
    vocab_data = (pd.DataFrame
                    .from_dict(vocab, orient='index', columns=['w']))
    sents = vocab_data.join(sents, how='left')
    sents = sents.sort_values('w')

    alpha = np.array(sents[['alpha_1', 'alpha_2', 'alpha_3']])
    alpha[np.isnan(alpha[:, 0]), :] = 1.

    return alpha


def process_data(config):
    '''
    Process the data.
    '''
    noun_map = parse_noun_map(config.map_fpath) if config.map_fpath else None
    examples, w_vocab, lemma_vocab = parse_collapsed_data(
        fpath=config.input_fpath,
        cols=config.input_cols,
        noun_map=noun_map,
        min_count=config.min_vocab_count,
    )
    
    if config.hold_out_data:
        # define train-test split
        np.random.seed(config.split_seed)
        examples['split'] = np.random.choice(
            ('test', 'train'),
            size=examples.shape[0],
            p=(1 - config.train_split, config.train_split),
        )
    if config.hold_out_nouns:
        # train-test split on nouns
        np.random.seed(config.split_seed)
        train_nouns = np.random.choice(
            noun_map.noun,
            size=int(noun_map.shape[0] * config.train_split),
            replace=False
        )
        examples['split'] = 'test'
        examples.loc[examples.noun.isin(train_nouns), 'split'] = 'train'
    
    if config.train_split == 1:
        # gen dummy "test" data
        pseudo_test = examples.iloc[0:1].copy().assign(split='test')
        examples = pd.concat([examples, pseudo_test], ignore_index=True)

    grouped_examples = pd.pivot_table(
        data=examples,
        values=['count'],
        columns='w',
        index=['fem', 'lemma', 'plural', 'split'],
        aggfunc=np.sum,
        fill_value=0.,
        dropna=True,
    )
    
    grouped_ex_train = grouped_examples.xs('train', level='split')
    grouped_ex_test = grouped_examples.xs('test', level='split')
    w_counts_train = grouped_ex_train.sum(0).values.astype(np.float32)    
    w_counts_test = grouped_ex_test.sum(0).values.astype(np.float32)

    assert(len(w_vocab) == len(w_counts_train) == len(w_counts_test))
    noun_counts = grouped_examples.xs('train', level='split').sum(1).reset_index()
    def noun_pct(col):
        return (noun_counts.groupby(col)
                           .sum()
                           .sort_values(col)
                           .apply(lambda x: x / x.sum(), axis=0)[0]
                           .values)

    # create graph data
    data = {
        'm': {
            'train': (
                np.log(w_counts_train, where=(w_counts_train!=0)) 
                - np.log(w_counts_train.sum()).reshape((-1, 1))
            ),
            'test': (
                np.log(w_counts_test, where=(w_counts_test!=0))
                - np.log(w_counts_test.sum()).reshape((-1, 1))
            ),
        },
        'w_prob': {
            'train': grouped_ex_train.apply(lambda x: x / x.sum(), axis=1).values,
            'test': grouped_ex_test.apply(lambda x: x / x.sum(), axis=1).values,
        },

        'fem': {
            'train': grouped_ex_train.index.get_level_values('fem').values,
            'test': grouped_ex_test.index.get_level_values('fem').values,
        },
        'lemma': {
            'train': grouped_ex_train.index.get_level_values('lemma').values,
            'test': grouped_ex_test.index.get_level_values('lemma').values,
        },
        'plural': {
            'train': grouped_ex_train.index.get_level_values('plural').values,
            'test': grouped_ex_test.index.get_level_values('plural').values,
        },

        'alpha': parse_sentiment_data(config.sent_fpath, w_vocab),
        'gender_freq': noun_pct('fem'),
        'lemma_freq': noun_pct('lemma'),
        'plural_freq': noun_pct('plural'),
        'noun_freq': noun_pct(['fem', 'lemma', 'plural']),
    }

    return data, w_vocab, lemma_vocab


'''
Model
'''

class ModelConfig:
    '''
    Class that stores configurations
    '''
    def __init__(self, **kwargs):
        self._config_dict = kwargs
    
    def __getattr__(self, name):
        '''
        When an <class_instance>.<attr> attempt is made, that instance's
        `__getattr__` method is called last in the lookup chain.

        This method explicitly transfers the key-value config paris stored in
        `self._config_dict` to the attributes of the instance (`self`) whenever
        they are accessed. The sister property `self.accessed_attrs` collects
        these attributes for later reference.
        '''
        try:
            val = self._config_dict[name]
        except KeyError:
            raise AttributeError(
                f'{__class__.__name__} object has no attribute {name}')

        # only happens once
        # `__getattr__` will not be called again for the same `name`
        if val is not None:  # only save attributes that aren't None!
            setattr(self, name, val)
        return val

def create_graph(config):
    '''
    Make the Tensorflow Graph
    '''
    tf.reset_default_graph()

    # trainable parameters
    eta_fem_sent = tf.get_variable(
        'eta_fem_sent',
        shape=(3, 2, 1, 1, config.w_size), # sentiment by gender by vocab
        dtype=tf.float32,
        initializer=config.sigma_initializer, # config.eta_initializer,
        constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
    )
    eta_lemma = tf.get_variable(
        'eta_lemma',
        shape=(1, 1, config.lemma_size, 1, config.w_size), # lemma by vocab
        dtype=tf.float32,
        initializer=config.sigma_initializer,
        constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
    )
    # eta_fem_lemma = tf.get_variable(
    #     'eta_fem_lemma',
    #     shape=(1, 2, config.lemma_size, 1, config.w_size), # lemma by vocab
    #     dtype=tf.float32,
    #     initializer=config.sigma_initializer,
    #     constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
    # )
    eta_plural = tf.get_variable(
        'eta_plural',
        shape=(1, 1, 1, 2, config.w_size), # plurality by vocab
        dtype=tf.float32,
        initializer=config.sigma_initializer,
        constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
    )

    sigma = tf.get_variable(
        'sigma', # for evaluating p(S | N)
        shape=(3, 2),
        dtype=tf.float32,
        initializer=config.sigma_initializer,
    )

    # inputs
    training = tf.placeholder(tf.bool) # train or test
    w_prob = tf.cond(
        training,
        lambda: tf.constant(config.w_prob['train'], dtype=tf.float32),
        lambda: tf.constant(config.w_prob['test'], dtype=tf.float32),
    )
    sparse_fem = tf.cond(
        training,
        lambda: tf.one_hot(config.fem['train'], depth=2, axis=-1, dtype=tf.float32),
        lambda: tf.one_hot(config.fem['test'], depth=2, axis=-1, dtype=tf.float32),
    )
    sparse_lemma = tf.cond(
        training,
        lambda: tf.one_hot(config.lemma['train'], depth=config.lemma_size, axis=-1, dtype=tf.float32),
        lambda: tf.one_hot(config.lemma['test'], depth=config.lemma_size, axis=-1, dtype=tf.float32),
    )
    sparse_plural = tf.cond(
        training,
        lambda: tf.one_hot(config.plural['train'], depth=2, axis=-1, dtype=tf.float32),
        lambda: tf.one_hot(config.plural['test'], depth=2, axis=-1, dtype=tf.float32),
    )
    m = tf.cond(
        training,
        lambda: tf.constant(config.m['train'].reshape(1, 1, 1, 1, config.w_size), dtype=tf.float32),
        lambda: tf.constant(config.m['test'].reshape(1, 1, 1, 1, config.w_size), dtype=tf.float32),
    )

    gender_freq = tf.constant(config.gender_freq.reshape(1, 2, 1, 1, 1), dtype=tf.float32)
    lemma_freq = tf.constant(config.lemma_freq.reshape(1, 1, -1, 1, 1), dtype=tf.float32)
    plural_freq = tf.constant(config.plural_freq.reshape(1, 1, 1, 2, 1), dtype=tf.float32)
    noun_freq = tf.constant(config.noun_freq.reshape(-1, 1), dtype=tf.float32)

    ## Calculating the likelihood ##
    # P(W | S, G, L, P)
    prob_w_given_sent_noun = tf.nn.softmax(
        # Sent, Gend, Lemma, Plural, Word
        m +                # [1 x 1 x 1 x 1 x |V|]
        eta_fem_sent +     # [S x G x 1 x 1 x |V|]
        eta_lemma +        # [1 x 1 x L x 1 x |V|]
        # eta_fem_lemma +    # [1 x G x L x 1 x |V|]
        eta_plural,        # [1 x 1 x 1 x P x |V|]
        axis=-1
    )
    # P(W | S, G=gender, L=lemma, P=plural)
    prob_w_given_sent_data = tf.einsum(
        'sglpv,ng,nl,np->nsv',
        prob_w_given_sent_noun,
        sparse_fem,
        sparse_lemma,
        sparse_plural,
    ) # [S x G x L x P x |V|][bs x G][bs x L][bs x P] -> [batch_size x S x |V|]

    prob_sent_given_gend = tf.nn.softmax(sigma, axis=0)
    prob_sent_given_gend_data = tf.matmul(
        sparse_fem,
        prob_sent_given_gend,
        transpose_b=True
    )
    # [batch_size x genders][genders x sents] -> [batch_size x sents]
    
    prob_w_given_data = tf.reduce_sum(
        prob_w_given_sent_data * 
        tf.reshape(prob_sent_given_gend_data, (-1, 3, 1)), # broadcast p(S|n)
        axis=1,
    ) # marginalize out S -> [batch_size x |V|]

    # cross entropy loss
    # cross_entropy = - tf.reduce_sum(tf.log(noun_freq * prob_w_given_data) * (noun_freq * w_prob), axis=1)
    cross_entropy = - tf.reduce_sum(tf.log(prob_w_given_data) * (w_prob), axis=1)

    ## Calculating the KL ##

    prob_sent_given_w = tf.constant(config.sent_freq.T, dtype=tf.float32)

    p_log_p = prob_sent_given_w * tf.log(prob_sent_given_w)

    prob_sent_noun_given_w = (
        prob_w_given_sent_noun # P(W | S, G, L, P)
        * tf.reshape(prob_sent_given_gend, (3, 2, 1, 1, 1)) # TODO: Check this is good
        * gender_freq # P(G)
        / tf.exp(m) # P(W) -- why does this division cause it no longer normalize?
    ) # [S x G x L x P x |V|]

    q = tf.reduce_sum(
        (
            prob_sent_noun_given_w
            / tf.reduce_sum(prob_sent_noun_given_w, axis=((0, 1, 2, 3)), keepdims=True)
        ), # renormalize
        axis=(1, 2, 3),
    )
    p_log_q = (
        prob_sent_given_w * # P(S | W)
        tf.log(q) # Q(S | W)
    )

    kl = tf.reduce_sum(p_log_p - p_log_q)

    ## l1 regularization on etas ##
    laplace_reg = tf.contrib.layers.l1_regularizer(scale=config.l1_reg_strength)
    laplace_penalty = tf.contrib.layers.apply_regularization(
        laplace_reg, [eta_fem_sent, eta_lemma, eta_plural], #eta_fem_lemma
    )
    loss = tf.reduce_mean(cross_entropy + config.kl_reg_strength * kl + laplace_penalty)
    optimizer = config.optimizer(learning_rate=config.learning_rate)
    min_op = optimizer.minimize(loss)

    components ={
        ### DEBUG
        'prob_sent_given_w': prob_sent_given_w,
        'prob_w_given_sent_noun': prob_w_given_sent_noun,
        'prob_sent_noun_given_w': prob_sent_noun_given_w,
        'prob_w_given_sent_data': prob_w_given_sent_data,
        'sparse_fem': sparse_fem,
        'sparse_lemma': sparse_lemma,
        'sparse_plural': sparse_plural,
        'q': q,
        'p_log_q': p_log_q,
        'p_log_p': p_log_p,

        ###
        'training': training,
        'kl': kl,
        'm': m,
        'likelihood': cross_entropy,
        'eta_fem_sent': eta_fem_sent,
        'eta_lemma': eta_lemma,
        # 'eta_fem_lemma': eta_fem_lemma,
        'eta_plural': eta_plural,
        'sigma': sigma,
        'loss': loss,
        'min_op': min_op,
        'init_op': tf.global_variables_initializer()
    }
    return components
    
def train(config_dict):
    '''
    Train the model
    '''
    config = ModelConfig(**config_dict)

    # if unspecified, get the (word, noun) count after which we only have the top 5000
    if config.min_vocab_count is None:
        config.min_vocab_count = get_count_cutoff(
            config.input_fpath,
            config.top_n_ws,
            config.input_cols,
        )

    # columns are (adjective/verb lemma, fem, plural, count)
    print('Processing data')
    data, w_vocab, lemma_vocab = process_data(config)

    config.w_size = len(w_vocab)
    config.lemma_size = len(lemma_vocab)
    config.m = data['m']
    config.w_prob = data['w_prob']
    config.fem = data['fem']
    config.lemma = data['lemma']
    config.plural = data['plural']

    config.sent_freq = data['alpha'] / data['alpha'].sum(1, keepdims=True)
    config.gender_freq = data['gender_freq']
    config.lemma_freq = data['lemma_freq']
    config.plural_freq = data['plural_freq']
    config.noun_freq = data['noun_freq']

    print('Building graph')
    graph_components = create_graph(config)

    ## training loop
    min_loss, prev_loss, i, flat_i = 1e10, 10., 0, 0
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        sess.run(graph_components['init_op'])

        while i < config.max_iter and flat_i <= config.max_flat_iter:
            i += 1
            # import ipdb
            # ipdb.set_trace()

            _, loss = sess.run(
                [graph_components['min_op'], graph_components['loss']],
                feed_dict={graph_components['training']: True}
            )               
            print(f'Evaluation at iter {i}: loss {loss:0.2f}\t\t\t', end='\r')
            diff = np.abs(loss - prev_loss) / prev_loss
            prev_loss = loss
            if diff < config.epsilon:
                flat_i += 1
            else:
                flat_i = 0 # reset the counter
            
            if loss < min_loss:
                min_loss = loss
                train_only = config.train_split == 1.
                results = sess.run({
                    'test_loss': graph_components['loss'], # N.B. uses train gend freq!
                    'test_likelihood': graph_components['likelihood'],
                    'm': graph_components['m'], 
                    'eta_fem_sent': graph_components['eta_fem_sent'],
                    'eta_lemma': graph_components['eta_lemma'],
                    # 'eta_fem_lemma': graph_components['eta_fem_lemma'],
                    'eta_plural': graph_components['eta_plural'],
                    'sigma': graph_components['sigma'],
                    'kl': graph_components['kl'],
                }, feed_dict={graph_components['training']: train_only})
    return w_vocab, lemma_vocab, data['alpha'], results, min_loss


CONFIG_DICT = {
    # data
    'input_fpath': './data/all-years-jj.txt',
    'map_fpath': './data/noun-map.txt',
    'sent_fpath': './models/vae_full_primedprior_softmax/sent_dict.csv',
    'processed_data_dir': './data/processed-noun_attrs',
    'input_cols': ['w', 'noun', 'count'], #['w', 'noun', 'lemma', 'fem', 'plural', 'count'],
    'save_processed_data': False,
    'load_processed_data': False,

    'train_split': 1.,
    'split_seed': 11235,
    'hold_out_nouns': True,
    'hold_out_data': False,

    'exclude_missing_sents': True,

    'min_vocab_count': 5,
    'top_n_ws': None, #10000,

    # hyperparams
    'eta_initializer': tf.initializers.zeros(),
    #'eta_initializer': lambda shape, dtype, partition_info: tf.cast(
    #     tf.distributions.Laplace(loc=0., scale=1.).sample(shape),
    #    dtype
    #),
    'sigma_initializer': tf.initializers.random_normal,

    'kl_reg_strength': 0.1,
    'l1_reg_strength': 0.00001,

    # training
    'max_iter': 25000,
    'epsilon': 0.000001,
    'max_flat_iter': 50, 
    
    # optimization
    'learning_rate': 0.001,
    'optimizer': tf.train.AdamOptimizer,
}
