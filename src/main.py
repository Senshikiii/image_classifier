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
# it'll be easy to edit stuff if the configs are at the top
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)
# 1e-3 --> 0.001
learning_rate = 1e-3 
batch_size 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()

















# why did I choose to use ReLU?? it's bcz sigmoid squishes everything btw 0 and 1, which causes a prob called vanishing gradients in deep networks, the nudges during backprop get tiny by the time they reach the early layers that those weghts barely update yadayayada
# we can use CrossEntropyLoss cuz instead of squaring it just measures how far the conf scroes are from the org answers, smarter ig? (uses a formula btw)
# optimizer is the thing which nudges the weights
# SGD, stochastic gradient descent, lr = learning rate + how big it is
# each epochs consists of two things, training loops and testing loops/validation loop

