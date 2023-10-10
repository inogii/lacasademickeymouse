# Native Libraries
import pprint as pp
import warnings
# Third-party Libraries
# General Utilities
import pandas as pd
import numpy as np
import datetime
# Machine Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import uniform, randint
# Visualization Libraries
from matplotlib import pyplot as plt


warnings.filterwarnings('ignore')

# global variables
train_size = 0.9
models = [LinearRegression(), ElasticNet(), Lasso(), Ridge(), KNeighborsRegressor(), RandomForestRegressor(), GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000), AdaBoostRegressor()]

price_min = 0
price_max = 0
price_mean = 0
price_std = 0

def minmax_norm(df, variables_reales):
    for variable in variables_reales:
        df[variable] = (df[variable] - df[variable].min()) / (df[variable].max() - df[variable].min())
    return df

def minmax_norm_price(df):
    global price_min, price_max

    price_min = df['Precio'].min()
    price_max = df['Precio'].max()
    df['Precio'] = (df['Precio'] - price_min) / (price_max - price_min)
    return df

def minmax_norm_price_inverse(np_array):
    global price_min, price_max
    return np_array * (price_max - price_min) + price_min

def zscore_norm(df, variables_reales):
    for variable in variables_reales:
        df[variable] = (df[variable] - df[variable].mean()) / df[variable].std()
    return df

def zscore_norm_price(df):
    global price_mean, price_std
    price_mean = df['Precio'].mean()
    price_std = df['Precio'].std()
    df['Precio'] = (df['Precio'] - price_mean) / price_std
    return df

def zscore_norm_price_inverse(np_array):
    global price_mean, price_std
    return np_array * price_std + price_mean


def one_hot_encoding(df, variables_categoricas):
    return pd.get_dummies(df, columns=variables_categoricas, dtype=np.int64)

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
    df = extract_postal_hierarchy(df)

    df['Reformada'] = df['FechaConstruccion'] != df['FechaReforma']
    df['Reformada'] = df['Reformada'].astype(int)

    current_year = datetime.datetime.now().year
    df['Aspect_Ratio'] = df['PerimParcela'] / df['Superficie']
    df['HighRating'] = df['RatingEstrellas'].apply(lambda x: 1 if x > 4 else 0)
    df['AgeOfHouse'] = current_year - df['FechaConstruccion']
    df['YearsSinceReform'] = current_year - df['FechaReforma']
    df['TotalRooms'] = df['Aseos'] + df['Habitaciones']
    df['AvgProximity'] = (df['ProxCarretera'] + df['ProxCallePrincipal'] + df['ProxViasTren']) / 3

    variables_reales = df.columns[df.dtypes == 'float64']
    variables_categoricas = df.dtypes[df.dtypes == 'object'].index
    variables_enteras = df.columns[df.dtypes == 'int64']

    variables_enteras = variables_enteras.drop(['Precio'])
    df = zscore_norm(df, variables_reales)
    df = zscore_norm(df, variables_enteras)
    df = zscore_norm_price(df)
    df = one_hot_encoding(df, variables_categoricas)

    corr = df.corr()
    umbral = 0.5
    # Encontrar características altamente correlacionadas
    caract_alta_correlación = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > umbral:
                colname = corr.columns[i]
                caract_alta_correlación.add(colname)

    print(caract_alta_correlación)
    df_filtered = df.drop(caract_alta_correlación, axis=1)
    df_filtered
    
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
    
class RegressionTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads, num_encoder_layers, dropout):
        super(RegressionTransformer, self).__init__()
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, 
                                                    nhead=num_heads, 
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                            num_layers=num_encoder_layers)
        # Final feed-forward layer to produce a single output for regression
        self.regressor = nn.Linear(feature_dim, 1)
    
    def forward(self, src):
        # src: (sequence_length, batch_size, feature_dim)
        # For non-sequential data, sequence_length will be 1.
        # Pass the input through the transformer encoder
        encoded_src = self.transformer_encoder(src)
        # If using non-sequential data with sequence_length=1, 
        # then squeeze the sequence_length dimension for simplicity
        if encoded_src.shape[0] == 1:
            encoded_src = encoded_src.squeeze(0)
        # Apply regressor for each item in the batch
        regression_output = self.regressor(encoded_src)
        
        return regression_output

def main():
    df = pd.read_csv('train.csv')
    df = dataset_preprocessing(df)

    train = df.sample(frac=train_size, random_state=1)
    test = df.drop(train.index)

    train = train[train['Precio'] < train['Precio'].quantile(0.95)]

    X_train, y_train = x_y_split(train, 'Precio')
    X_test, y_test = x_y_split(test, 'Precio')

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42
    )
    
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Learning rate scheduler (optional but can help with convergence)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15, threshold=0.0001, threshold_mode='rel')
    # Training loop
    epochs = 100
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
    #y_pred = y_pred_tensor.numpy()
    y_pred = y_pred_tensor.detach().numpy()
    y_pred = zscore_norm_price_inverse(y_pred)
    y_test = zscore_norm_price_inverse(y_test.values.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    visualize_test(y_test, y_pred, ax, "Transformer")
    plt.show()

if __name__ == "__main__":
    main()

