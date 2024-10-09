data_struct = load("data_train.mat");
label_struct = load("label_train.mat");

X = data_struct.data_train()
y = label_struct.label_train()
label = num2str(y)
% label = num2cell(label)
% tabulate(label_struct)

% tabulate(label_struct)