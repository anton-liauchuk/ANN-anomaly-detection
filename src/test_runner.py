import csv

from numpy import argmax
from pybrain import FullConnection, SoftmaxLayer, KohonenMap
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules.softsign import SoftSignLayer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet

INP = 41
TARGET = 5
MAX_EPOCHS = 100
RETRAIN_EPOCHS = 100


def read_records_kdd():
    ds = ClassificationDataSet(INP, TARGET, TARGET)
    with open('../input/compare_results.txt', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ds.addSample(row[:INP], row[INP:46])
    return ds


class ClassifNetwork:
    def __init__(self):
        pass

    @staticmethod
    def build_model_rnn_mlp():
        network = buildNetwork(INP, 12, 10, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = read_records_kdd()
        print ds
        trainer = BackpropTrainer(network, ds)
        # trainer = RPropMinusTrainer(network, dataset=ds, verbose=True, learningrate=0.02, momentum=0.88)
        trainer.trainUntilConvergence(maxEpochs=3, verbose=True)
        for i in range(MAX_EPOCHS):
            print "Progress: %d/%d \r" % (i, MAX_EPOCHS)
            # sys.stdout.flush()
            trainer.train()
        NetworkWriter.writeToFile(network, '../models/model_class.xml')
        print network.activate(
            [0, 0, 0, 0, 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0.00, 0.00, 0.00, 0.00, 1.00,
             0.00, 0.00, 9, 9, 1.00, 0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00])


ds = SupervisedDataSet(2, target=1)
# ds.addSample((1, 2), (5.69, 8))
network = buildNetwork(2, 1, outclass=KohonenMap)
# reccon = FullConnection(network['hidden0'], network['in'])
# network.addRecurrentConnection(reccon)
network.sortModules()
print network.activate((1, 768))
print network.activate((2,3))
print ((-1, -23))
# print ds
