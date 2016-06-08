import csv

import numpy as np

replacements_dict = []
training_attack_types = '../input/training_attack_types.txt'
TEST_INPUT_FILE = '../input/test_data.txt'


def change_symbolic_data(input_file, output_file):
    """For machine learning needed change symbolic input data."""
    protocol_types = {}
    services = {}
    flags = {}
    formatted_lines = []
    with open(input_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if not row[1].isdigit():
                if row[1] in protocol_types:
                    row[1] = protocol_types[row[1]]
                else:
                    protocol_types[row[1]] = str(len(protocol_types))
                    row[1] = protocol_types[row[1]]
            if not row[2].isdigit():
                if row[2] in services:
                    row[2] = services[row[2]]
                else:
                    services[row[2]] = str(len(services))
                    row[2] = services[row[2]]
            if not row[3].isdigit():
                if row[3] in flags:
                    row[3] = flags[row[3]]
                else:
                    flags[row[3]] = str(len(flags))
                    row[3] = flags[row[3]]
            if len(row) == 42:
                encoded_attack = find_attack_encoding(training_attack_types, row[41].replace('.', ''))
                row.remove(row[41])
                for number in encoded_attack:
                    row.append(number)
            formatted_lines.append(row)
    write_formatted_data(formatted_lines, output_file)
    replacements_dict.append(protocol_types)
    replacements_dict.append(services)
    replacements_dict.append(flags)
    print replacements_dict


def cut_learning_data(input_file, output_file):
    training_normal_inputs = []
    training_dos_inputs = []
    training_u2r_inputs = []
    training_r2l_inputs = []
    training_probe_inputs = []
    with open(input_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            attack = row[41:46]
            if attack == ['0', '0', '0', '0', '1'] and len(training_normal_inputs) != 1500:
                training_normal_inputs.append(row)
            if attack == ['1', '0', '0', '0', '0'] and len(training_dos_inputs) != 3571:
                training_dos_inputs.append(row)
            if attack == ['0', '1', '0', '0', '0'] and len(training_u2r_inputs) != 37:
                training_u2r_inputs.append(row)
            if attack == ['0', '0', '1', '0', '0'] and len(training_r2l_inputs) != 278:
                training_r2l_inputs.append(row)
            if attack == ['0', '0', '0', '1', '0'] and len(training_probe_inputs) != 800:
                training_probe_inputs.append(row)
        print len(training_normal_inputs)
        # training_inputs = [training_normal_inputs, training_r2l_inputs, training_dos_inputs, training_u2r_inputs,
        #                     training_probe_inputs]
        # training_inputs.append(dict(training_normal_inputs))
        # training_inputs = [training_probe_inputs]
        # print len(training_inputs)
        add_training_data(training_normal_inputs, output_file)
        add_training_data(training_probe_inputs, output_file)
        add_training_data(training_dos_inputs, output_file)
        add_training_data(training_r2l_inputs, output_file)
        add_training_data(training_u2r_inputs, output_file)


def cut_dos_set(input_file, output_file):
    training_dos_inputs = []
    with open(input_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            attack = row[41:46]
            if attack == ['1', '0', '0', '0', '0']:
                training_dos_inputs.append(row)
    add_training_data(training_dos_inputs, output_file)


def find_attack_encoding(input_file, type_attack):
    with open(input_file, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if row[0] == type_attack:
                return str(row[2]).split(',')


def write_formatted_data(formatted_lines, output_file):
    with open(output_file, 'wb') as outfile:
        wtr = csv.writer(outfile)
        wtr.writerows(formatted_lines)


def cut_test_data():
    lines = []
    with open('../input/kddcup.txt') as infile:
        for line in infile:
            lines.append(line)
    with open('cut_data.txt', 'w') as outfile:
        for line in lines:
            outfile.write(line)

            # write_formatted_data(cut_lines, '../input/cut_data.txt')


def add_training_data(inputs, file):
    with open(file, 'ab') as outfile:
        wtr = csv.writer(outfile)
        wtr.writerows(inputs)


# create_replacements_dictionary('../input/kddcup.txt')
# create_training_input_data('../input/formatted_data.txt', '../input/training_inputs.txt')


# cut_test_data()


def normalize_data(input_file, output_file):
    """Normalizing data"""
    formatted_lines = []
    with open(input_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            vet = np.array(map(float, row[:41]))
            norm = [float(i) / sum(vet) for i in vet]
            norm.extend([int(numeric_string) for numeric_string in row[41:46]])
            formatted_lines.append(norm)
    write_formatted_data(formatted_lines, output_file)
    print 'implement in feature'

normalize_data('../input/formatted_training.txt', '../input/normalize_training.txt')