import numpy as np
import tensorflow as tf
import pandas as pd


def load_model(path, session):

    from rbm.rbm import RBM
    from rbm.learning.constant_learning_rate import ConstantLearningRate

    rbm = RBM(
        visible_size=117,
        # hidden_size=10,
        hidden_size=6,
        learning_rate=ConstantLearningRate(0.05)
    )

    rbm.load(session, path)

    return rbm


def generate_input_vector(plugins_actives):
    visible_size = 117
    v = np.zeros((visible_size, 1))

    for plugin in plugins_actives:
        v[plugin][0] = 1

    return v

with tf.Session() as session:
    model = load_model("experiments/model/{}/rbm.ckpt".format(1), session)

    v0 = generate_input_vector([1, 3])
    h0 = model.sample_h_given_v(v0)
    v1_probabilities = model.P_v_given_h(h0)

    v1_probabilities = session.run(v1_probabilities)
    probabilities = pd.Series(v1_probabilities.T[0])

    print(probabilities.sort_values(ascending=False).head(10))


'''
P(plugin | 1,3)
plugin  p(plugin|1,3)
82     0.040000
78     0.040000
37     0.022727
54     0.010811
105    0.010000
93     0.009524
41     0.008475
73     0.006757

RBM com 100 epoch
28    0.770785
64    0.343765
30    0.244711
60    0.213227
24    0.165979
68    0.159112
73    0.145676
53    0.141537
23    0.138630
71    0.127165

RBM com 500 epoch
27     0.374272
30     0.354911
63     0.243221
24     0.240410
23     0.205305
92     0.172354
64     0.162931
25     0.144788
38     0.124545
111    0.118677
'''