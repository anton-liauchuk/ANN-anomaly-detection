import csv

from pybrain import FullConnection
from pybrain.structure.modules.softsign import SoftSignLayer
from pybrain.supervised import BackpropTrainer
from pybrain.tools.customxml import NetworkReader
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.datasets import SupervisedDataSet

INP = 41
TARGET = 5
TRAIN_EPOCHS = 3000
RETRAIN_EPOCHS = 5


class AnomalyDetectionNetwork:
    def __init__(self):
        pass

    @staticmethod
    def read_training_set(input_file):
        ds = SupervisedDataSet(INP, TARGET)
        with open(input_file, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                ds.addSample(row[:INP], row[INP:46])
        return ds

    @staticmethod
    def build_model_1():
        network = buildNetwork(INP, 12, 10, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = AnomalyDetectionNetwork.read_training_set('../input/norm.txt')
        trainer = BackpropTrainer(network, ds)
        trainer.trainUntilConvergence(maxEpochs=TRAIN_EPOCHS, verbose=True)
        NetworkWriter.writeToFile(network, '../models/model_1.xml')

    @staticmethod
    def build_model_2():
        network = buildNetwork(INP, 12, 40, 10, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = AnomalyDetectionNetwork.read_training_set('../input/normalize_training.txt')
        normalize_test = AnomalyDetectionNetwork.read_training_set('../input/normalize_test.txt')
        trainer = BackpropTrainer(network, learningrate=0.1)
        trainer.trainUntilConvergence(maxEpochs=RETRAIN_EPOCHS, verbose=True, trainingData=ds, validationData=normalize_test)
        NetworkWriter.writeToFile(network, '../models/model_2.xml')

    @staticmethod
    def retrain_model_1():
        network = NetworkReader.readFrom('../models/model_6_new.xml')
        ds = AnomalyDetectionNetwork.read_training_set('../input/normalize_training.txt')
        normalize_test = AnomalyDetectionNetwork.read_training_set('../input/normalize_test.txt')
        trainer = BackpropTrainer(network, learningrate=0.3)
        trainer.trainUntilConvergence(maxEpochs=RETRAIN_EPOCHS, verbose=True, trainingData=ds, validationData=normalize_test)
        NetworkWriter.writeToFile(network, '../models/model_6_new.xml')

    @staticmethod
    def build_model_3():
        network = buildNetwork(INP, 36, 36, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = AnomalyDetectionNetwork.read_training_set('../input/normalize_training.txt')
        normalize_test = AnomalyDetectionNetwork.read_training_set('../input/normalize_test.txt')
        trainer = BackpropTrainer(network, learningrate=0.1)
        trainer.trainUntilConvergence(maxEpochs=RETRAIN_EPOCHS, verbose=True, trainingData=ds, validationData=normalize_test)
        NetworkWriter.writeToFile(network, '../models/model_6.xml')

    def write_model(output_file):
        print 'write'

    def read_model(input_file):
        print 'read'
