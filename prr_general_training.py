import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import geopandas as gpd
import csv
import random
from torch.utils.data import Dataset, DataLoader
from time import sleep
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils import class_weight
import seaborn as sn
import matplotlib.pyplot as plt
from pickle import dump
from sklearn.metrics import f1_score
import statsmodels.api as sm
import xgboost as xgb

def file_names(path):
    dirs = os.listdir(path)
    return dirs


def load_data(path, file):
    path = os.path.join(path, file)
    with open(path, newline='') as f:
        data = csv.reader(f)
        data = list(data)
    return data


class CoverageData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        #y_predicted = self.linear(x)
        return y_predicted
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv1d(1, 16, 3)
        self.conv1 = nn.Conv1d(1, 16, 1)
        # self.pool = nn.MaxPool1d(2, 2)
        self.pool = nn.MaxPool1d(1, 1)
        # self.conv2 = nn.Conv1d(16, 8, 3)
        self.conv2 = nn.Conv1d(16, 8, 1)
        # self.fc1 = nn.Linear(288, 120)  # Adjust the input size based on your input shape (1, 151)
        self.fc1 = nn.Linear(2024, 512)
        # self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(84, 1)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x = torch.reshape(x, (-1, 1, 151))
        x = torch.reshape(x, (-1, 1, 253))
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = torch.reshape(x, (-1, 1))
        return x

    
def BCELoss_customized(weights,pred,label):
    pred = torch.clamp(pred,min = 1e-7,max = 1 - 1e-7)
    loss = - weights[1]*label*torch.log(pred) - (1 - label)*weights[0]*torch.log(1 - pred)
    return torch.mean(loss)


def f_score(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f_score


def sigmoid(z):
    """
    Compute the sigmoid function for the input z.

    Parameters:
    z : array-like or scalar
        Input values.

    Returns:
    ndarray or scalar
        Sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


def check_list(lst):
    # Check if there are any negative numbers in the list
    if any(num < 0 for num in lst):
        return 0
    else:
        return 1


def round_to_zero(lst):
    return [0 if num < 0.9 else 1 for num in lst]


def combine_numpy_arrays_to_csv(arr1, arr2, output_file):
    """
    Combines two NumPy arrays of arrays into a CSV file. Each row in the CSV contains elements from a sub-array in the 
    first array followed by the corresponding sub-array from the second array.

    Args:
    arr1 (numpy.ndarray): The first array of arrays.
    arr2 (numpy.ndarray): The second array of arrays.
    output_file (str): The path to the output CSV file.
    """
    # Ensure both arrays have the same length
    assert len(arr1) == len(arr2), "Both arrays must have the same number of elements"
    
    # Combine the arrays along the horizontal axis
    combined_array = np.hstack((arr1, arr2))

    # Save the combined array to a CSV file
    np.savetxt(output_file, combined_array, delimiter=',', fmt='%f')


train_cities = ['brooklyn']
all_cities = ['ithaca', 'geneva', 'brooklyn']
type = "test"

combined_X_train = []
combined_y_train = []

combined_X_test = []
combined_y_test = []
for city in train_cities:
    X_path = "/mnt/data1/aar245/" + city + "_results/" + city + "_real_success_and_fail_size_250/"
    y_path = "/mnt/data1/aar245/geojsons/" + city + "_real_success_and_fail.geojson"

    files = file_names(X_path)
    random.seed(10)
    random.shuffle(files)

    y_data = gpd.read_file(y_path)

    batch_size = 32

    dataset = []
    for idx, file in enumerate(files):
        X_data = load_data(X_path, file)
        X_data = [x for xs in X_data for x in xs]
        X_data = [float(i) for i in X_data]
        i = file.split('.')
        i = int(i[0])
        X_data.append(y_data.success[i])
        X_data.append(y_data.ele_tr[i])
        X_data.append(y_data.ele_gw[i])
        X_data.append(np.log(y_data.dist[i]/1000))
        X_data.append(y_data.rssi[i])
        X_data.append(y_data.lon_tx[i])
        X_data.append(y_data.lat_tx[i])
        dataset.append(X_data)

        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int((idx+1)/len(files)*20), (idx+1)/len(files)*100))
        sys.stdout.flush()

    dataset_train = dataset[0:round(0.8*len(dataset))]
    dataset_test = dataset[round(0.8*len(dataset)):]

    lon_tx = [i[-2] for i in dataset_test]
    lat_tx = [i[-1] for i in dataset_test]
    dist = [np.exp(i[-4])*1000 for i in dataset_test]
    gw_ele = [i[-5] for i in dataset_test]
    tx_ele = [i[-6] for i in dataset_test]
    success = [i[-7] for i in dataset_test]
    los_test = [sum(x > 0 for x in i[0:-7])/len(i[0:-7]) for i in dataset_test]

    X = [i[0:-7] for i in dataset]
    y = [i[-7] for i in dataset]
    dist_x = [i[-4] for i in dataset]
    hb = [np.abs(i[-6] - i[-5]) for i in dataset]

    print("min dist: ", np.min(dist_x))
    print("max_dist: ", np.max(dist_x))
    print("mean_dist: ", np.mean(dist_x))
    print("min_dist_reg: ", np.min(np.exp(dist_x)))
    print("max_dist_reg: ", np.max(np.exp(dist_x)))

    X = [X[i] + [dist_x[i], np.log(hb[i])] for i in range(len(X))]

    idcs = []
    for idx, val in enumerate(hb):
        if val <= 0:
            idcs.append(idx)

    X = [value for index, value in enumerate(X) if index not in idcs]

    y = [value for index, value in enumerate(y) if index not in idcs]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

    combined_X_train.append(X_train)
    combined_X_test.append(X_test)
    combined_y_train.append(y_train)
    combined_y_test.append(y_test)


##############################################################
# For missing cities when training/testing on 2 cities

missing = list(set(all_cities) - set(train_cities))

if len(missing) > 0:
    for i in missing:
        X_path = "/mnt/data1/aar245/" + i + "_results/" + i + "_real_success_and_fail_size_250/"
        #y_path = "/mnt/data1/aar245/geojsons/geneva_real_success_and_fail_new.geojson"
        y_path = "/mnt/data1/aar245/geojsons/" + i + "_real_success_and_fail.geojson"
        #y_path = "/mnt/data1/aar245/geojsons/brooklyn_real_success_and_fail.geojson"

        files = file_names(X_path)
        random.seed(10)
        random.shuffle(files)

        y_data = gpd.read_file(y_path)

        batch_size = 32

        dataset = []
        for idx, file in enumerate(files):
            X_data = load_data(X_path, file)
            X_data = [x for xs in X_data for x in xs]
            X_data = [float(i) for i in X_data]
            i = file.split('.')
            i = int(i[0])
            X_data.append(y_data.success[i])
            X_data.append(y_data.ele_tr[i])
            X_data.append(y_data.ele_gw[i])
            X_data.append(np.log(y_data.dist[i]/1000))
            ###########################################
            X_data.append(y_data.rssi[i])
            ###########################################
            X_data.append(y_data.lon_tx[i])
            X_data.append(y_data.lat_tx[i])
            dataset.append(X_data)

            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int((idx+1)/len(files)*20), (idx+1)/len(files)*100))
            sys.stdout.flush()

        dataset_train = dataset[0:round(0.8*len(dataset))]
        dataset_test = dataset[round(0.8*len(dataset)):]

        lon_tx = [i[-2] for i in dataset_test]
        lat_tx = [i[-1] for i in dataset_test]
        dist = [np.exp(i[-4])*1000 for i in dataset_test]
        gw_ele = [i[-5] for i in dataset_test]
        tx_ele = [i[-6] for i in dataset_test]
        success = [i[-7] for i in dataset_test]
        los_test = [sum(x > 0 for x in i[0:-7])/len(i[0:-7]) for i in dataset_test]

        X = [i[0:-7] for i in dataset]
        y = [i[-7] for i in dataset]
        dist_x = [i[-4] for i in dataset]
        hb = [np.abs(i[-6] - i[-5]) for i in dataset]

        print("min dist: ", np.min(dist_x))
        print("max_dist: ", np.max(dist_x))
        print("mean_dist: ", np.mean(dist_x))
        print("min_dist_reg: ", np.min(np.exp(dist_x)))
        print("max_dist_reg: ", np.max(np.exp(dist_x)))

        X = [X[i] + [dist_x[i], np.log(hb[i])] for i in range(len(X))]

        idcs = []
        for idx, val in enumerate(hb):
            if val <= 0:
                idcs.append(idx)

        X = [value for index, value in enumerate(X) if index not in idcs]

        y = [value for index, value in enumerate(y) if index not in idcs]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

        combined_X_test.append(X_test)
        combined_y_test.append(y_test)

##############################################################
out = ''
for i in train_cities:
    out+= i + '_'

X_train = np.concatenate(combined_X_train)
y_train = np.concatenate(combined_y_train)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
with open(out + 'scaler_model.pkl', 'wb') as f:
    dump(sc, f)

y_train = [int(i) for i in y_train]
y_train = np.array(y_train)
y_train = np.reshape(y_train, (y_train.shape[0], 1))

weights = [
    len(y_train) / (len(np.unique(y_train))*np.where(y_train == 0)[0].size),
    len(y_train) / (len(np.unique(y_train))*np.where(y_train == 1)[0].size)
]

freq_weights = []
for i in y_train:
    if i == 0:
        freq_weights.append(weights[0])
    else:
        freq_weights.append(weights[1])

X_train = sm.add_constant(X_train)

if type == 'xg':
    X_train = xgb.DMatrix(X_train, label=y_train, weight=freq_weights)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 10, #20
        'eta': 0.3,
        'seed': 10
    }

    num_round = 1000 # train Geneva and Brooklyn: 1000 #5000
    model = xgb.train(params, X_train, num_round)

    model.save_model(out + "xgboost_combined_250x1_with_d_h.json")
    
    for i in range(len(combined_y_test)):
        X_test = sc.transform(combined_X_test[i])
        X_test = sm.add_constant(X_test)
        y_test = [int(j) for j in combined_y_test[i]]
        y_test = np.array(y_test)
        y_test = np.reshape(y_test, (y_test.shape[0], 1))
        X_test = xgb.DMatrix(X_test, label=y_test)
       
        y_pred = model.predict(X_test)

        results = pd.DataFrame([i for i in zip(y_test, y_pred, lon_tx, lat_tx, gw_ele, tx_ele, success, dist, los_test)], columns = ["prr_real", "prr_pred", "lon_tx", "lat_tx", "gw_ele", "tx_ele", "success", "dist", "los"])
        #results.to_csv("geneva_new_" + str(struct) + ".csv", index = False)
        results.to_csv(out + "xgboost_combined_test_" + all_cities[i] + "_250x1_with_d_h.csv", index = False)

        # Convert DMatrix to NumPy array
        X_test = X_test.get_data()
        X_test = X_test.toarray()
        combine_numpy_arrays_to_csv(X_test, y_test, out + "xgboost_combined_" + all_cities[i] + "_test_data.csv")


if type == 'lr':
    log_reg = sm.GLM(y_train, X_train, freq_weights = freq_weights, family=sm.families.Binomial()).fit()
    print(log_reg.summary())

    log_reg.save(out + "combined_250x1_with_d_h.pickle")

    for i in range(len(combined_y_test)):
        X_test = sc.transform(combined_X_test[i])
        y_test = [int(j) for j in combined_y_test[i]]
        y_test = np.array(y_test)
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        X_test = sm.add_constant(X_test)
        y_pred = log_reg.predict(X_test)
        results = pd.DataFrame([i for i in zip(y_test, y_pred, lon_tx, lat_tx, gw_ele, tx_ele, success, dist, los_test)], columns = ["prr_real", "prr_pred", "lon_tx", "lat_tx", "gw_ele", "tx_ele", "success", "dist", "los"])
        results.to_csv(out + "combined_test_" + all_cities[i] + "_250x1_with_d_h.csv", index = False)
        combine_numpy_arrays_to_csv(X_test, y_test, out + "combined_" + all_cities[i] + "_test_data.csv")


if type == 'cnn':
    X_train = torch.from_numpy(np.array(X_train).astype('float32'))
    y_train = torch.from_numpy(np.array(y_train).astype('float32'))
    train_dataset = CoverageData(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size)
    
    learning_rate = 5e-4
    model = Net()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    temp_y = y_train.view(-1).tolist()
    
    class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(temp_y), y = temp_y)
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    criterion = nn.MSELoss()

    y_train = y_train.view(y_train.shape[0], 1)

    num_epochs = 10000

    best_loss = np.inf
    patience_counter = 0
    patience = 10

    for k in range(len(combined_y_train)):
        X_test = sc.transform(combined_X_test[k])
        y_test = [int(j) for j in combined_y_test[k]]
        y_test = np.array(y_test)
        y_test = np.reshape(y_test, (y_test.shape[0], 1))
        X_test = sm.add_constant(X_test)

        X_test = torch.from_numpy(np.array(X_test).astype('float32'))
        y_test = torch.from_numpy(np.array(y_test).astype('float32'))
        test_dataset = CoverageData(X_test, y_test)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size)
        y_test = y_test.view(y_test.shape[0], 1)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (features, labels) in enumerate(train_loader):
                y_predicted = model(features)
                labels = labels.view(labels.shape[0], 1)
                loss = BCELoss_customized(class_weights, y_predicted, labels)
                running_loss += loss.item() * features.size(0)
                #loss = criterion(y_predicted, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
            epoch_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(test_loader.dataset)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(model.state_dict(), out + "cnn_combined_250x1_with_d_h.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    for q in range(len(combined_y_test)):
        X_test = sc.transform(combined_X_test[q])
        y_test = [int(j) for j in combined_y_test[q]]
        y_test = np.array(y_test)
        y_test = np.reshape(y_test, (y_test.shape[0], 1))
        X_test = sm.add_constant(X_test)

        X_test = torch.from_numpy(np.array(X_test).astype('float32'))
        y_test = torch.from_numpy(np.array(y_test).astype('float32'))
        test_dataset = CoverageData(X_test, y_test)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size)
        y_test = y_test.view(y_test.shape[0], 1)
        total_acc = 0
        count = 0
        true = []
        reg_pred = []
        class_pred = []
        with torch.no_grad():
            for i, (features, labels) in enumerate(test_loader):
                y_predicted = model(features)
                reg_pred.append(y_predicted.cpu().detach().numpy())
                y_predicted = torch.round(y_predicted)
                acc = accuracy_score(labels, y_predicted)
                total_acc += acc
                count += 1
                true.append(labels.cpu().detach().numpy())
                class_pred.append(y_predicted.cpu().detach().numpy())
            total_acc = total_acc / count
            print(f'final_accuracy = {total_acc:.4f}')

        true = [x for xs in true for x in xs]
        reg_pred = [x for xs in reg_pred for x in xs]
        class_pred = [x for xs in class_pred for x in xs]

        results = pd.DataFrame([i for i in zip(true, reg_pred, lon_tx, lat_tx, gw_ele, tx_ele, success, dist, los_test)], columns = ["prr_real", "prr_pred", "lon_tx", "lat_tx", "gw_ele", "tx_ele", "dist", "success", "los"])
        results.to_csv(out + "cnn_combined_test_" + all_cities[q] + "_250x1_with_d_h.csv", index = False)

        combine_numpy_arrays_to_csv(X_test, y_test, out + "cnn_combined_" + all_cities[q] + "_test_data.csv")

if type == 'test':
    print('test')