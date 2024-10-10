data_struct = load("data/data_train.mat");
label_struct = load("data/label_train.mat");
test_struct = load("data/data_test.mat");

X = data_struct.data_train();
y = label_struct.label_train();
X_test = test_struct.data_test();
label = num2str(y);