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
RETRAIN_EPOCHS = 100


class AnomalyDetectionNetwork:
    def __init__(self):
        pass

    @staticmethod
    def read_records_kdd():
        ds = SupervisedDataSet(INP, TARGET)
        with open('../input/norm.txt', 'rb') as f:
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
        ds = AnomalyDetectionNetwork.read_records_kdd()
        trainer = BackpropTrainer(network, ds)
        trainer.trainUntilConvergence(maxEpochs=TRAIN_EPOCHS, verbose=True)
        NetworkWriter.writeToFile(network, '../models/model_1.xml')

    @staticmethod
    def build_model_2():
        network = buildNetwork(INP, 12, 40, 10, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = AnomalyDetectionNetwork.read_records_kdd()
        trainer = BackpropTrainer(network, ds)
        NetworkWriter.writeToFile(network, '../models/model_2.xml')

    @staticmethod
    def retrain_model_1():
        network = NetworkReader.readFrom('../models/model_1.xml')
        ds = AnomalyDetectionNetwork.read_records_kdd()
        trainer = BackpropTrainer(network, ds)
        trainer.trainUntilConvergence(maxEpochs=RETRAIN_EPOCHS, verbose=True)
        NetworkWriter.writeToFile(network, '../models/model_1_new.xml')

    @staticmethod
    def build_model_2():
        print 'model_2'

    @staticmethod
    def build_model_3():
        print 'model_3'

    def write_model(output_file):
        print 'write'

    def read_model(input_file):
        print 'read'
