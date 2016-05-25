from pybrain.tools.customxml import NetworkReader
import numpy as np
import csv

from network_builder import AnomalyDetectionNetwork


# from prepare_input_tools import change_attack_names_for_learning

# AnomalyDetectionNetwork.read_records_kdd()
# change_attack_names_for_learning()
# AnomalyDetectionNetwork.build_model_rnn_mlp()

test_results = {(1, 0, 0, 0, 0): [0, 0, 0],
                (0, 1, 0, 0, 0): [0, 0, 0],
                (0, 0, 1, 0, 0): [0, 0, 0],
                (0, 0, 0, 1, 0): [0, 0, 0],
                (0, 0, 0, 0, 1): [0, 0, 0]}

#
# def compare_output_and_attack(input_file, output_file):
#     with open('../input/formatted_data.txt', 'rb') as f:
#         reader = csv.reader(f, delimiter=',')
#         for row in reader:
#             result_network = net.activate(row[0:40])
#             attack = row[41:46]
#             if attack != ['0', '0', '0', '0', '1']:
#                 result_network
#                 # add check for detect atack or no


# add check for needed_atack or no

def test_compare():
    net = NetworkReader.readFrom('../models/model_1.xml')
    with open('../input/compare_results.txt', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            attack = map(int, row[41:46])
            input_parameters = row[0:41]
            # print map(float, input_parameters)
            result_network = net.activate(map(float, input_parameters))
            low_values_indices = result_network < np.max(result_network)
            result_network[low_values_indices] = 0
            result_network[result_network.argmax()] = 1
            if attack == [0, 0, 0, 0, 1]:
                if map(int, result_network.tolist()) == attack:
                    test_results[tuple(attack)][1] += 1
            else:
                if map(int, result_network.tolist()) == attack:
                    test_results[tuple(attack)][0] += 1
                    test_results[tuple(attack)][1] += 1
            test_results[tuple(attack)][2] += 1
    with open('../input/results.txt', 'wb') as outfile:
        wtr = csv.writer(outfile)
        for key, val in test_results.items():
            wtr.writerow([key, val])


def format_output(output):
    # np.around(output)
    low_values_indices = output < np.max(output)
    output[low_values_indices] = 0
    print output


# net = NetworkReader.readFrom('../models/model_1.xml')

# test_compare(net.activate(
#     [0, 2, 9, 0, 1032, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 511, 511, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00,
#      0.00, 255, 255, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00]))
test_compare()