from model import TorchModel
import torch 


def train():
    n = TorchModel()
    n.batch_train()
    torch.save(n.state_dict(), "params.pt")

def load_test():
    n = TorchModel()
    n.load_state_dict(torch.load("params.pt"))
    n.test_data()

def train_and_test():
    n = TorchModel()
    n.batch_train()
    n.test_data()

def test_neighbours():
    n = TorchModel()
    n.load_state_dict(torch.load("params.pt"))
    n.prepare_neighbours()
    n.test_data()

#train()
# load_test()
# train_and_test()
test_neighbours()
