from io import BytesIO
import os
import pandas as pd
from sklearn.metrics import mean_squared_log_error
import subprocess
import sys

INP_FILE = 'sample_in.csv'
OUT_FILE = 'sample_out.csv'


def test (cmd, file, real):
    """
    Tests the output of a ML command.

    @param cmd: the command to evaluate.
    @param file: the datafile to introduce.
    @param real: the real output.
    @return a score of the tested command.
    """
    output = subprocess.check_output([sys.executable, cmd, file])
    csvStr = BytesIO(output)
    data = pd.read_csv(csvStr)
    return mean_squared_log_error(real.Precio, data.Precio, squared=False)


if __name__ == '__main__':
    real = pd.read_csv(OUT_FILE)
    for folder in filter(lambda f : os.path.isdir(f), os.listdir('.')):
        try:
            score = test(f'{folder}/main.py', INP_FILE, real)
            print(f'{folder}: {score}')
        except Exception as e:
            print(f'Ha ocurrido un error con {folder}: {e}')