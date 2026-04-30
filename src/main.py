import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image
from torch.utils.data import Dataset

# all the imports


class SafeImageFolder(datasets.ImageFolder):           # we already know how to use imagefolder, but this class helps us skip an image and move forward if the current one is crashed (skip and try next one)
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            return self.__getitem__((index + 1) % len(self))



# these are the hyerparameters (adj params which lets you optimize your nn)
learning_rate = 1e-3
batch_size = 64
epochs = 70


train_transform = transforms.Compose([  # chaining multiple transforms, running them in order
    transforms.Resize((64, 64)), # resizes images to 64x64 
    transforms.RandomHorizontalFlip(), # flips em horinzontally (randomly)
    transforms.RandomRotation(10), # rotates itt (to 10 degrees) 
    transforms.ToTensor(), # tranforms into tensor (0 - 1)
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
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 3 input channels, 32 filters, each 3x3
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # shrinks an image by half??
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 feat maps, produce 64 feature maps?
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # shrink again by half
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # turn 50% of the neurons off during training to prevent overfitting
            nn.Linear(512, 2) # only two outputs (cat or dog)
        )
    def forward(self, x):
        x = self.conv_layers(x) # detect features
        return self.fc_layers(x) # classify based on features

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = CNN().to(device)
print(model)

# using loss fn + optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # total number of images
    model.train()
    for batch, (X, y) in enumerate(dataloader):     # loop through batches
        X, y = X.to(device), y.to(device) # move data to the same device as model
        pred = model(X)
        loss = loss_fn(pred, y) # measures how wrong the nn is
        loss.backward() # figures out nudges via backprop
        optimizer.step() # applies the nudges
        optimizer.zero_grad() # resets gradients for next batch
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]") # prints progress


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset) # total test images
    num_batches = len(dataloader) # number of batches
    test_loss, correct = 0, 0 
    with torch.no_grad(): # don't calc the gradients
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) # forward pass
            test_loss += loss_fn(pred, y).item() # combine the losses??
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches    # avg loss
    correct /= size # accuracy
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

for t in range(epochs):
    print(f"Epochs {t+1} \n -----------------------")
    training_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    print("Done")

torch.save(model.state_dict(), 'model_weights_cats_dogs.pth')




