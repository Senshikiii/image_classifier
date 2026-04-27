import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image
from torch.utils.data import Dataset

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            return self.__getitem__((index + 1) % len(self))



# these are the hyerparameters (adj params which lets you optimize your nn)
learning_rate = 1e-3
batch_size = 64
epochs = 10


train_transform = transforms.Compose([
    transforms.Resize((64, 64)), # resizes images 
    transforms.RandomHorizontalFlip(), # flips em horinzontally
    transforms.RandomRotation(10), # rotates itt
    transforms.ToTensor(), # tranforms into tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # it means scaling input data to a specific range or distribution?? 
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# the data part, using ImageFolder btw
training_data = SafeImageFolder(root="data_2/train", transform=train_transform)
test_data = SafeImageFolder(root="data_2/test", transform=test_transform)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# the fun part ig? 
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = CNN().to(device)
print(model)

# using loss fn + optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

for t in range(epochs):
    print(f"Epochs {t+1} \n -----------------------")
    training_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    print("Done")

torch.save(model.state_dict(), 'model_weights_cats_dogs.pth')




