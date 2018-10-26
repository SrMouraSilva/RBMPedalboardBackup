import pandas as pd


def read_data(path):
    return pd.read_csv(path, sep=",", index_col=['index'])


def treat_input(bag_of_plugins):
    """
    Consider only pedalboards with more then 3 distinct plugins
    """
    # Convert only zero and one
    bag_of_plugins = ((bag_of_plugins > 0) * 1)

    # Remove guitar_patches.id column
    del bag_of_plugins['id']
    # Remove never unused plugin
    # > bag_of_plugins.T[bag_of_plugins.sum() == 0]
    # 9
    #del bag_of_plugins['9']

    # Zero all that will disconsider
    # BOMB(9) and None(107)
    bag_of_plugins['9'] = 0
    bag_of_plugins['107'] = 0

    # Consider only pedalboards with more then 3 distinct plugins
    bag_of_plugins = bag_of_plugins[bag_of_plugins.T.sum() > 3]
    return bag_of_plugins


def train(dataset, batch_size=10, epochs=100, persist=False):
    """
    # Batch_size = 10 or 100
    # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """
    import time

    import tensorflow as tf

    from rbm.learning.constant_learning_rate import ConstantLearningRate
    from rbm.rbm import RBM
    from rbm.regularization.regularization import L1Regularization, L2Regularization
    from rbm.train.task.beholder_task import BeholderTask
    from rbm.train.task.persistent_task import PersistentTask
    from rbm.train.task.rbm_mensurate_task import RBMMeasureTask
    from rbm.train.task.summary_task import SummaryTask
    from rbm.train.trainer import Trainer

    tf.set_random_seed(42)

    total_elements, size_element = dataset.shape

    rbm = RBM(
        visible_size=size_element,
        #hidden_size=10,
        hidden_size=6,
        #regularization=L1Regularization(0.01),
        #regularization=L2Regularization(0.01),
        learning_rate=ConstantLearningRate(0.05)
    )

    trainer = Trainer(rbm, dataset, batch_size=batch_size)
    trainer.stopping_criteria.append(lambda current_epoch: current_epoch > epochs)

    log = "experiments/logs/{}/{}".format(batch_size, time.time())

    trainer.tasks.append(RBMMeasureTask())
    trainer.tasks.append(SummaryTask(log=log))
    trainer.tasks.append(BeholderTask(log='experiments/logs'))

    if persist:
        trainer.tasks.append(PersistentTask(path="experiments/model/{}/rbm.ckpt".format(batch_size)))

    trainer.train()


# Treinar
bag_of_plugins = read_data('data/pedalboard-plugin-bag-of-words.csv')
bag_of_plugins = treat_input(bag_of_plugins)
train(bag_of_plugins, persist=True, batch_size=1, epochs=100)

# Isso aqui vai ser usado para verificar se as probabilidades batem
#Counselor(bag_of_plugins).suggest(['1', '3'])


# Gerar amostras
# Verificar se a probabilidade que deu bate com a do dataset
