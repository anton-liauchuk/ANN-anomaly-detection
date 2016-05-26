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
MAX_EPOCHS = 500


class AnomalyDetectionNetwork:
    def __init__(self):
        pass

    @staticmethod
    def read_records_kdd():
        ds = SupervisedDataSet(INP, TARGET)
        index = 0
        # with open('../input/formatted_data.txt') as f:
        #     fdata = [line.rstrip() for line in f]
        # while index < 6000:
        #     array = fdata[index].replace('.\n', '')
        #     records = array.rsplit(',')
        #     ds.addSample(records[:INP], records[INP:46])
        #     index += 1
        with open('../input/training_inputs.txt', 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                ds.addSample(row[:INP], row[INP:46])
        return ds

    @staticmethod
    def build_model_rnn_mlp():
        network = buildNetwork(INP, 12, 10, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = AnomalyDetectionNetwork.read_records_kdd()
        # trainer = BackpropTrainer(network, ds, learningrate=0.02, momentum=0.88)
        trainer = BackpropTrainer(network, ds)
        # trainer.trainUntilConvergence()
        for i in range(MAX_EPOCHS):
            print "Progress: %d/%d \r" % (i, MAX_EPOCHS)
            # sys.stdout.flush()
            trainer.train()
        NetworkWriter.writeToFile(network, '../models/model_1.xml')
        print network.activate([0, 1, 11, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 4, 0.00, 0.00, 0.00, 0.00, 0.12,
                0.09, 0.00, 255, 48, 0.19, 0.20, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00])

    @staticmethod
    def build_model_2():
        network = buildNetwork(INP, 12, 40, 10, TARGET, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        ds = AnomalyDetectionNetwork.read_records_kdd()
        # trainer = BackpropTrainer(network, ds, learningrate=0.02, momentum=0.88)
        trainer = BackpropTrainer(network, ds)
        # trainer.trainUntilConvergence()
        for i in range(MAX_EPOCHS):
            print "Progress: %d/%d \r" % (i, MAX_EPOCHS)
            # sys.stdout.flush()
            trainer.train()
        NetworkWriter.writeToFile(network, '../models/model_2.xml')
        print network.activate(
            [0, 1, 11, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 4, 0.00, 0.00, 0.00, 0.00, 0.12,
             0.09, 0.00, 255, 48, 0.19, 0.20, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00])


    @staticmethod
    def retrain_model_1():
        network = NetworkReader.readFrom('../models/model_1.xml')
        ds = AnomalyDetectionNetwork.read_records_kdd()
        # trainer = BackpropTrainer(network, ds, learningrate=0.02, momentum=0.88)
        trainer = BackpropTrainer(network, ds)
        # trainer.trainUntilConvergence()
        for i in range(MAX_EPOCHS):
            print "Progress: %d/%d \r" % (i, MAX_EPOCHS)
            # sys.stdout.flush()
            trainer.train()
        NetworkWriter.writeToFile(network, '../models/new_model_1.xml')


    @staticmethod
    def build_model_mlp_for_attack():
        print 'model_2'

    @staticmethod
    def build_model_expert():
        print 'model_3'

    def write_model(output_file):
        print 'write'

    def read_model(input_file):
        print 'read'



