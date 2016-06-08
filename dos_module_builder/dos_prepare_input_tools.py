import csv

from prepare_input_tools import add_training_data, normalize_data


def cut_dos_training_set(input_file, output_file):
    training_dos_inputs = []
    number_examples_1 = 0
    number_examples_2 = 0
    number_examples_3 = 0
    with open(input_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[0:4] == ['0', '2', '9', '0'] and number_examples_1 != 2000:
                training_dos_inputs.append(row)
                number_examples_1 += 1
            if row[0:4] == ['0', '0', '11', '4'] and number_examples_2 != 1000:
                training_dos_inputs.append(row)
                number_examples_2 += 1
            if row[0:4] == ['0', '0', '11', '2'] and number_examples_3 != 571:
                number_examples_3 += 1
                training_dos_inputs.append(row)
    add_training_data(training_dos_inputs, output_file)


def change_dos_training(new_dos):
    training_inputs = []
    with open('../input/formatted_training.txt', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[41:46] != ['1', '0', '0', '0', '0']:
                training_inputs.append(row)
    with open(new_dos, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            training_inputs.append(row)
    add_training_data(training_inputs, '../input/formatted_training_new.txt')

change_dos_training('../dos_module_builder/dos_formatted_training.txt')
# cut_dos_training_set('../dos_module_builder/dos_formatted_test.txt', '../dos_module_builder/dos_formatted_training.txt')
# normalize_data('../dos_module_builder/dos_formatted_training.txt', '../dos_module_builder/dos_normalize_training.txt')
