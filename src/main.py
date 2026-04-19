import torch
from torch import nn

from torch.utils.data import DataLoader

from torchvision import datasets

from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# shuffle so the network doesn't memorize the order
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# don't shuffle so the results will be consistent
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

class NeuralNetwork(nn.Module):  # i mean it's jus a class 
    def __init__(self): # we pretty much put all our layers here
        super().__init__()  # this thing runs pytorch's setup before our setup runs
        self.flatten = nn.Flatten() # stretches the img grid into a long line of numbers
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), 
        )
    def forward(self, x): # x is the img input btw
        x = self.flatten(x)
        return self.layers(x)

model = NeuralNetwork()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# why did I choose to use ReLU?? it's bcz sigmoid squishes everything btw 0 and 1, which causes a prob called vanishing gradients in deep networks, the nudges during backprop get tiny by the time they reach the early layers that those weghts barely update yadayayada

