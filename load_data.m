data_struct = load("data/data_train.mat");
label_struct = load("data/label_train.mat");

X = data_struct.data_train();
y = label_struct.label_train();
label = num2str(y);

newDataPoint = [5.0, 3.5, 1.5, 0.2, 0.5];
% label = num2cell(label)
% tabulate(label_struct)

% tabulate(label_struct)