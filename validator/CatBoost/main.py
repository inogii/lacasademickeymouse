# Trabajo de Aprendizaje Automático de lacasademickeymouse
# Juan Misas Higuera
# Fernando Martín San Bruno
# Íñigo González-Varas Vallejo
# Eduardo de la Vega Fernández


import sys
import pandas as pd
import datetime
import numpy as np
import pickle
                        
import os


# Obtener la ruta absoluta del script en ejecución
script_path = os.path.abspath(__file__)
# Obtener el directorio en el que se encuentra el script
script_dir = os.path.dirname(script_path)

dir = script_dir + os.sep + "modelo.pickle"

with open(dir, "rb") as handler:
    model = pickle.load(handler)


def extract_postal_hierarchy(df):
    df['CP'] = df['CP'].astype(str)
    df['postal_group'] = df['CP'].str[0]
    df['region'] = df['CP'].str[0:2]
    return df


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


def predict(df):
    output = pd.DataFrame()
    output['Id'] = df.index
    y_pred = model.predict(df)
    output['Precio'] = np.expm1(y_pred).round(2)
    output['Precio'] = np.clip(output['Precio'], 0.5**5, 1.5*10**6)
    return output

def dataset_preprocessing(df):
    
    df.set_index('Id', inplace=True)
    df.drop(['AguaCorriente', 'GasNatural', 'FosaSeptica', 'Piscina', 'Plan', 'PerimParcela'], axis=1, inplace=True)

    df = extract_postal_hierarchy(df)

    df['SinGaraje'] = df['Garaje'].apply(lambda x: 1 if x == 0 else 0)
    df['3plantas'] = df['Plantas'].apply(lambda x: 1 if x == 3 else 0)

    current_year = datetime.datetime.now().year

    df['AgeOfHouse'] = current_year - df['FechaConstruccion']
    df['YearsSinceReform'] = current_year - df['FechaReforma']

    df['aseos+hab*rating'] = (0.7*df['Aseos'] + 0.3*df['Habitaciones']) * df['RatingEstrellas']
    df['synth4'] = np.log1p(df['aseos+hab*rating'] * df['Superficie'])
    df['synth5'] = np.log1p(df['aseos+hab*rating'] * df['Estado'])
    df['synth6'] = np.log1p(df['Estado'] * df['Superficie'])
    df['synth7'] = np.log1p(df['aseos+hab*rating'] * df['Superficie'] * df['Estado'])

    df.drop(['FechaConstruccion', 'FechaReforma', 'Garaje', 'Formato', 'TipoDesnivel', 'Desnivel', 'Situacion', 'Plantas', 'PAU', 'Vallada', 'Callejon', 'CallePavimentada', 'Aseos', 'Habitaciones'], axis=1, inplace=True)

    variables_reales = df.columns[df.dtypes == 'float64']
    variables_categoricas = df.dtypes[df.dtypes == 'object'].index
    variables_enteras = df.columns[df.dtypes == 'int64']

    df = one_hot_encoding(df, variables_categoricas)

    df.drop(['Tipo_Chalet individual', 'CatParcela_Residencial tipo 2', 'CatParcela_Residencial unifamiliar', 'CP_50012', 'CP_50018', 'CP_60645', 'CP_61704', 'CP_62451'], axis=1, inplace=True)

    df = zscore_norm(df, variables_reales)
    df = zscore_norm(df, variables_enteras)
    
    columns = ['Id','3plantas','AgeOfHouse','CP_50010','CP_50011','CP_50014','CP_50015','CP_50017','CP_60061','CP_60118','CP_60406','CP_60646','CP_60706','CP_60936','CP_61528','CP_61615','CP_61705','CP_61874','CP_62040','CP_62447','CP_62801','CatParcela_Residencial especial','CatParcela_Residencial tipo 1','CatParcela_Terciario','Estado','Precio','ProxCallePrincipal','ProxCarretera','ProxViasTren','RatingEstrellas','SinGaraje','Superficie','Tipo_Casa baja','Tipo_Chalet adosado','Tipo_Dúplex','Tipo_Piso alto','YearsSinceReform','aseos+hab*rating','postal_group_5','postal_group_6','region_50','region_60','region_61','region_62','synth4','synth5','synth6','synth7']

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Eliminamos las columnas que sobran
    df = df[columns]

    return df



if __name__ == "__main__":
    file_name = sys.argv[1]

    df = pd.read_csv(file_name)

    df = dataset_preprocessing(df)

    output = predict(df)

    # Writes the output.
    print('Id,Precio')
    for r in output.itertuples():
        print(f'{r.Id},{r.Precio}')

