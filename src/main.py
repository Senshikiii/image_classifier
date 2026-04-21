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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)
# 1e-3 --> 0.001
learning_rate = 1e-3 
batch_size = 64
epochs = 5


def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        #figures out how much each weight contributed to the loss
        # actually applies the nudges
        # resets the graidents
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches  = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 14
for t in range(epochs):
    print(f"Epoch + {t+1} \n---------------------------")
    training_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    print("Done")



# btw pred = model snippet does the forward pass thing, loss function does its job

# why did I choose to use ReLU?? it's bcz sigmoid squishes everything btw 0 and 1, which causes a prob called vanishing gradients in deep networks, the nudges during backprop get tiny by the time they reach the early layers that those weghts barely update yadayayada
# we can use CrossEntropyLoss cuz instead of squaring it just measures how far the conf scroes are from the org answers, smarter ig? (uses a formula btw)
# optimizer is the thing which nudges the weights
# SGD, stochastic gradient descent, lr = learning rate + how big it is
# each epochs consists of two things, training loops and testing loops/validation loop

