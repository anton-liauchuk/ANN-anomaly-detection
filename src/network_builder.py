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
TRAIN_EPOCHS = 1000
RETRAIN_EPOCHS = 50


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
        ds = AnomalyDetectionNetwork.read_training_set('../input/norm.txt')
        trainer = BackpropTrainer(network, ds)
        NetworkWriter.writeToFile(network, '../models/model_2.xml')

    @staticmethod
    def retrain_model_1():
        network = NetworkReader.readFrom('../models/model_1.xml')
        ds = AnomalyDetectionNetwork.read_training_set('../dos_module_builder/dos_normalize_training.txt')
        trainer = BackpropTrainer(network, ds)
        trainer.trainUntilConvergence(maxEpochs=RETRAIN_EPOCHS, verbose=True)
        NetworkWriter.writeToFile(network, '../models/model_3.xml')

    @staticmethod
    def build_model_2():
        print 'model_2'

    @staticmethod
    def build_model_3():
        network = buildNetwork(INP, 12, 10, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = AnomalyDetectionNetwork.read_training_set('../input/normalize_training.txt')
        trainer = BackpropTrainer(network, ds)
        trainer.trainUntilConvergence(maxEpochs=TRAIN_EPOCHS, verbose=True)
        NetworkWriter.writeToFile(network, '../models/model_3.xml')

    def write_model(output_file):
        print 'write'

    def read_model(input_file):
        print 'read'
