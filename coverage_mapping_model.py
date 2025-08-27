import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Is this the correct approach for generating train/val/test datasets?
# What is X? X is dataset created from make_geodata + log(distance)
# What is y? y comes from geopandas df received/no received binary
# When testing using Geneva data, how do I create linksfile and rasterfile? I will create linksfile. Rasterfile comes from dsm
# Is there a DTM for Geneva? Do I need DTM? DTM is DSM
# How do you get elevation data? I get elevation from DSM
# Geneva correct projection: utm = "EPSG:26918"  # https://epsg.io/26918

# ~2000 examples collected in Geneva today. Try training a model using the following features:
    # Log(distance), transmitter elevation, receiver elevation, line of site variable (positives/total)
    # Plotting coverage will use probabilities at each longitude/latitude
# 7980 successes, 89352 failures. Should there be more balanced classes?
    # 1) Throw out obvious failures first. Short range failures are more interesting. Sort by distance btw gateway and transmission
    # 2) Use sample weights in pytorch. Scale each observation by the weight when loss is computed
# What is meant by log(distance)? Is distance measured btw transmission and gateway?
# What other features can be included? Accuracy is ~29% right now (I believe)
    # Elevation, transmitted power, spreading factor. All features should be those that are available where data has not yet been transmitted.
# Formulating optimization problem: How to optimize gateway placement from this?

X_path = "/mnt/Expansion/aar245/test/coverage_mapping/antwerp_data/antwerp_output_geo_condensed/"
y_path = "/mnt/Expansion/aar245/test/coverage_mapping/antwerp_data/antwerp_output_nogeo/antwerp_test.geojson"

# X_path = "/mnt/Expansion/aar245/test/coverage_mapping/antwerp_data/geneva_750/"
# y_path = "/mnt/Expansion/aar245/test/coverage_mapping/antwerp_data/geneva/geneva_march23.geojson"

# X_path = "/mnt/Expansion/aar245/test/coverage_mapping/antwerp_data/ithaca_new/"
# y_path = "/mnt/Expansion/aar245/test/coverage_mapping/antwerp_data/geneva/ithaca.geojson"

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
        self.fc1 = nn.Linear(24, 12)
        # self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(12, 6)
        # self.fc3 = nn.Linear(84, 1)
        self.fc3 = nn.Linear(6, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x = torch.reshape(x, (-1, 1, 151))
        x = torch.reshape(x, (-1, 1, 3))
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = torch.sigmoid(x)
        x = torch.reshape(x, (-1, 1))
        return x

        
def BCELoss_customized(weights,pred,label):
    pred = torch.clamp(pred,min = 1e-7,max = 1 - 1e-7)
    loss = - weights[1]*label*torch.log(pred) - (1 - label)*weights[0]*torch.log(1 - pred)
    return torch.mean(loss)


files = file_names(X_path)
random.shuffle(files)
#files.sort(key = len)

# geneva_files = file_names(X_path_geneva)
# random.shuffle(geneva_files)

# train_files = files[0:round(0.8*len(files))]
# test_files = files[round(0.8*len(files)):]

y_data = gpd.read_file(y_path)

# y_data_geneva = gpd.read_file(y_path_geneva)

batch_size = 32

dataset = []
for idx, file in enumerate(files):
    X_data = load_data(X_path, file)
    X_data = [x for xs in X_data for x in xs]
    X_data = [float(i) for i in X_data]
    i = file.split('.')
    i = int(i[0])
    X_data.append(y_data.success[i])
    #X_data.append(y_data.ele_tr[i])
    X_data.append(y_data.ele_tx[i])
    X_data.append(y_data.ele_gw[i])
    X_data.append(np.log(y_data.dist[i]))
    ###########################################
    X_data.append(y_data.rssi[i])
    ###########################################
    X_data.append(y_data.lon_tx[i])
    X_data.append(y_data.lat_tx[i])
    dataset.append(X_data)

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int((idx+1)/len(files)*20), (idx+1)/len(files)*100))
    sys.stdout.flush()

# dataset_geneva = []
# for idx, file in enumerate(geneva_files):
#     X_data = load_data(X_path_geneva, file)
#     X_data = [x for xs in X_data for x in xs]
#     X_data = [float(i) for i in X_data]
#     i = file.split('.')
#     i = int(i[0])
#     X_data.append(np.log(y_data_geneva.dist[i]))
#     X_data.append(y_data_geneva.success[i])
#     X_data.append(y_data_geneva.lon_tx[i])
#     X_data.append(y_data_geneva.lat_tx[i])
#     dataset_geneva.append(X_data)

#     sys.stdout.write('\r')
#     sys.stdout.write("[%-20s] %d%%" % ('='*int((idx+1)/len(geneva_files)*20), (idx+1)/len(geneva_files)*100))
#     sys.stdout.flush()

dataset_train = dataset[0:round(0.8*len(dataset))]
dataset_test = dataset[round(0.8*len(dataset)):]
lon_tx = [i[-2] for i in dataset_test]
lat_tx = [i[-1] for i in dataset_test]
dist = [np.exp(i[-4]) for i in dataset_test]
gw_ele = [i[-5] for i in dataset_test]
tx_ele = [i[-6] for i in dataset_test]
success = [i[-7] for i in dataset_test]
los_test = [sum(x > 0 for x in i[0:-7])/len(i[0:-7]) for i in dataset_test]

X = [i[0:-7] for i in dataset]
y = [i[-7] for i in dataset]
dist_x = [i[-4] for i in dataset]

# X_geneva = [i[0:-1] for i in dataset_geneva]
# y_geneva = [i[-1] for i in dataset_geneva]

# X = X + X_geneva
# y_total = y + y_geneva

# X_train = X
# y_train = y

# X_test = X_geneva
# y_test = y_geneva

# X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size = 0.5, random_state = 1234)

los = [sum(x > 0 for x in i)/len(i) for i in X]

X = [[dist_x[i], los[i], los[i]*dist_x[i]] for i in range(len(X))]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
# dump(sc, open('scaler_test.pkl', 'wb'))
X_test = sc.transform(X_test)

X_train = torch.from_numpy(np.array(X_train).astype('float32'))
X_test = torch.from_numpy(np.array(X_test).astype('float32'))
y_train = torch.from_numpy(np.array(y_train).astype('float32'))
y_test = torch.from_numpy(np.array(y_test).astype('float32'))

train_dataset = CoverageData(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size)

test_dataset = CoverageData(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size)

n_inputs = np.asarray(X_train).shape[1]
n_outputs = 1

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

model = LogisticRegression(n_inputs, n_outputs)
# model = Net()

# learning_rate = 0.001
learning_rate = 1e-6
#learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(y), y = y)
class_weights = torch.tensor(class_weights,dtype=torch.float)
#criterion = nn.BCELoss(weight = class_weights, reduction = 'mean')
# criterion = nn.BCELoss()
criterion = nn.MSELoss()

num_epochs = 20000
# num_epochs = 10000
tracker = 0
track_loss = 1000000000
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        y_predicted = model(features)
        labels = labels.view(labels.shape[0], 1)
        loss = BCELoss_customized(class_weights, y_predicted, labels)
        #loss = criterion(y_predicted, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    if loss.item() < track_loss:
        track_loss = loss.item()
        tracker = 0

    if loss.item() > track_loss:
        tracker += 1

    if tracker > 5:
        break

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

labels = ['Unsuccessful', 'Successful']

true = [x for xs in true for x in xs]
reg_pred = [x for xs in reg_pred for x in xs]
class_pred = [x for xs in class_pred for x in xs]

#print('f1_score: ', f1_score(true, class_pred, zero_division=1.0))
# cf_matrix = confusion_matrix(true, class_pred)

# df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = labels,
#                      columns = labels)

# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.xlabel('Prediction')
# plt.ylabel('True')
# plt.title("LR Train/Test Geneva")
# plt.savefig('3features_geneva_LR_march12_rssi.png')

# for param in model.parameters():
#   print(param.data)

torch.save(model.state_dict(), "/mnt/Expansion/aar245/test/coverage_mapping/proppy/examples/antwerp/models/3features_antwerp_LR_april3_prr.pth")

results = pd.DataFrame([i for i in zip(lon_tx, lat_tx, true, reg_pred, gw_ele, tx_ele, dist, success, los_test)], columns = ["lon_tx", "lat_tx", "true_rssi", "prediction_rssi", "gw_ele", "tx_ele", "dist", "success", "los"])
results.to_csv("3features_antwerp_LR_april3_prr.csv", index = False)