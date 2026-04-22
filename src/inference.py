import torch
from torch import nn
from torchvision import transforms
from PIL import Image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


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
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
image = Image.open("/Users/yashwinsenshiki/cool_images/numbers/images (1).jpeg")
image = transform(image)


# unsqueezing adds a batch, so instead of this:[1, 28, 28] it becomes this -> [1, 1, 28, 28]
image = image.unsqueeze(0)
with torch.no_grad():
    pred = model(image)
    predicted = pred.argmax(1).item()
    print(f"Predicted digit: {predicted}")


