import os
import sys

from train_utils.train import train_model
from test_utils.test import test_model

curpath = os.path.abspath(os.path.dirname(__file__))
print(curpath)
sys.path.append(curpath)


# if __name__ == "__main__":
    # train_model()
    # test_model()

