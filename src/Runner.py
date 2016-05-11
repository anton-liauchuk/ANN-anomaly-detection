from pybrain.tools.customxml import NetworkReader

from ann import AnomalyDetectionNetwork

# from prepare_input_tools import change_attack_names_for_learning

# AnomalyDetectionNetwork.read_records_kdd()
# change_attack_names_for_learning()
# AnomalyDetectionNetwork.build_network()
net = NetworkReader.readFrom('../models/model_1.xml')
print net.activate(
    [0,2,9,0,1032,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,511,511,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,255,1.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00])
