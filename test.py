from model import TorchModel
import torch 


def train():
    n = TorchModel()
    n.load_data()
    n.batch_train()
    torch.save(n.state_dict(), "params.pt")

def load_test():
    n = torch.load("params.pt")
    n.test_data()

train()
load_test()
