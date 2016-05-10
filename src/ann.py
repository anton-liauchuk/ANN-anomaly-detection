from pybrain import FullConnection
from pybrain.structure.modules.softsign import SoftSignLayer
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.datasets import SupervisedDataSet

INP = 41
TARGET = 5


class AnomalyDetectionNetwork:
    def __init__(self):
        pass

    @staticmethod
    def read_records_kdd():
        ds = SupervisedDataSet(41, 5)
        # lines = [line.rstrip('.') for line in open('records.txt')]
        # with open('test.txt') as f:
        #     fdata = [line.rstrip() for line in f]
        index = 0
        error = 0
        # while index < 100 and error == 1:
        #     try:
        #         array = fdata[index].replace('\n', '')
        #         records = array.rsplit(',')
        #         print records[:41]
        #         print records[41:46]
        #         ds.addSample(records[:41], records[41:46])
        #         index += 1
        #     except ValueError:
        #         print fdata[index]
        #         print index
        #         error = 1
        # with open('records.txt') as f:
        with open('records.txt') as f:
            fdata = [line.rstrip() for line in f]
        while index < 2:
            array = fdata[index].replace('.\n', '')
            records = array.rsplit(',')
            ds.addSample(records[:41], records[41:46])
            index += 1
        return ds
        # ds._convertToOneOfMany( bounds=[0,1] )
        # with open('records.txt') as f:
        #    lines = f.readlines()
        # lines = open('records.txt', 'r')
        # print lines.read()

    @staticmethod
    def build_network():
        network = buildNetwork(41, 12, 10, 5, bias=True, outclass=SoftSignLayer, recurrent=True)
        reccon = FullConnection(network['hidden0'], network['in'])
        network.addRecurrentConnection(reccon)
        network.sortModules()
        print network.activate(
            [

                0, 0, 0, 0, 239, 486, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0.00, 0.00, 0.00, 0.00,
                1.00, 0.00, 0.00, 19, 19, 1.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00])
        ds = AnomalyDetectionNetwork.read_records_kdd()
        trainer = BackpropTrainer(network, ds, learningrate=0.02, momentum=0.88)
        # for epoch in range(0,700):
        trainer.trainUntilConvergence(ds, 1190)
        print network.activate(
            [

                0, 0, 0, 0, 239, 486, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0.00, 0.00, 0.00, 0.00,
                1.00, 0.00, 0.00, 19, 19, 1.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00])
