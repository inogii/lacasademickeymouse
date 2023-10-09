import pandas as pd
import numpy as np
import pprint as pp
import pgeocode as pg
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import uniform, randint
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')

# global variables
train_size = 0.9
models = [LinearRegression(), ElasticNet(), Lasso(), Ridge(), KNeighborsRegressor(), RandomForestRegressor(), GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000), AdaBoostRegressor()]

def minmax_norm(df, variables_reales):
    for variable in variables_reales:
        df[variable] = (df[variable] - df[variable].min()) / (df[variable].max() - df[variable].min())
    return df

def zscore_norm(df, variables_reales):
    for variable in variables_reales:
        df[variable] = (df[variable] - df[variable].mean()) / df[variable].std()
    return df

def one_hot_encoding(df, variables_categoricas):
    return pd.get_dummies(df, columns=variables_categoricas, dtype=np.int64)

def label_encoding(df, variables_categoricas):
    for variable in variables_categoricas:
        df[variable] = df[variable].astype('category').cat.codes
    return df

def extract_postal_hierarchy(df):
    df['CP'] = df['CP'].astype(str)
    df['postal_group'] = df['CP'].str[0]
    df['region'] = df['CP'].str[:3]
    df['specific_location'] = df['CP']
    return df

def x_y_split(df, target):
    return df.drop(target, axis=1), df[target]

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_pred

def visualize_test(y_test, y_pred, ax, model_name):
    paired = sorted(list(zip(y_test, y_pred)))
    y_test_sorted, y_pred_sorted = zip(*paired)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape= mean_absolute_percentage_error(y_test, y_pred)
    # print in scientific notation format
    print('MSE: {:.2e}'.format(mse))
    print('MAE: {:.2e}'.format(mae))
    print('MAPE: {:.2e}'.format(mape))
    # plot y_test and y_pred values to visualize the model performance
    num_range = np.arange(0, len(y_test))
    ax.plot(num_range, y_test_sorted, label='y_test', marker='*', color='blue')
    ax.plot(num_range, y_pred_sorted, label='y_pred', marker='.', color='red')
    ax.set_title(f'y_test vs y_pred {model_name}')
    ax.legend()


def dataset_preprocessing(df):

    df.index = df['Id']
    df.drop(['Id', 'AguaCorriente', 'GasNatural', 'FosaSeptica'], axis=1, inplace=True)
    df.dropna(inplace=True)

    variables_reales = df.columns[df.dtypes == 'float64']
    variables_categoricas = df.dtypes[df.dtypes == 'object'].index
    variables_enteras = df.columns[df.dtypes == 'int64']

    variables_enteras = variables_enteras.drop(['Precio', 'CP'])

    df = minmax_norm(df, variables_reales)
    df = minmax_norm(df, variables_enteras)
    df = one_hot_encoding(df, variables_categoricas)
    df = extract_postal_hierarchy(df)

    df['Reformada'] = df['FechaConstruccion'] != df['FechaReforma']
    df['Reformada'] = df['Reformada'].astype(int)
    
    return df

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(m.bias)

# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), # First hidden layer with 64 neurons
            nn.ReLU(),
            nn.Dropout(0.5),          # Dropout layer to prevent overfitting
            nn.Linear(64, 32),        # Second hidden layer with 32 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)          # Output layer
        )

    def forward(self, x):
        return self.layers(x)

class SophisticatedNet(nn.Module):
    def __init__(self, input_dim):
        super(SophisticatedNet, self).__init__()
        
        # Define hidden layer sizes
        hidden1 = 512
        hidden2 = 256
        hidden3 = 128
        hidden4 = 64

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)  # Batch normalization after first layer
        self.dropout1 = nn.Dropout(0.5)     # Dropout layer

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(hidden3, hidden4)
        self.bn4 = nn.BatchNorm1d(hidden4)
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(hidden4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)
        return x


def main():
    df = pd.read_csv('train.csv')
    df = dataset_preprocessing(df)

    train = df.sample(frac=train_size, random_state=1)
    test = df.drop(train.index)

    X_train, y_train = x_y_split(train, 'Precio')
    X_test, y_test = x_y_split(test, 'Precio')

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42
    )
    # Convert your data to PyTorch tensors
    # Convert your data to PyTorch tensors
    # Convert your data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)


    # Create DataLoader for your data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = X_train.shape[1]
    model = SophisticatedNet(input_dim)
    model.apply(initialize_weights)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.5)

    # Learning rate scheduler (optional but can help with convergence)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30, threshold=1000000)
    # Training loop
    epochs = 400
    val_loss_list = []
    train_loss_list = []
    for epoch in range(epochs):
        lr_now = get_lr(optimizer=optimizer)
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        scheduler.step(val_loss)  # adjust learning rate based on the scheduler
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader):.6f}, Validation Loss: {val_loss:.6f}, LR: {lr_now}")

        train_loss_list.append(running_loss/len(train_loader))
        val_loss_list.append(val_loss)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss_list)
    ax.plot(val_loss_list)
    plt.show()

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    visualize_test(y_test, y_pred, ax, "SophisticatedNet")
    plt.show()

if __name__ == "__main__":
    main()

