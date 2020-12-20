import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import matplotlib.pyplot as plt


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid * 100


class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SIRENLayer, self).__init__()
        self.in_features = in_features
        self.lin = nn.Linear(in_features, out_features)
        # self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.lin.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

    def forward(self, x):
        out = self.lin(x)
        out = torch.sin(out)
        return out


class SIREN(nn.Module):
    def __init__(self, features_of_layers):
        super(SIREN, self).__init__()

        self.siren_layers = nn.ModuleList()
        for i in range(len(features_of_layers) - 1):
            layer_i = SIRENLayer(features_of_layers[i], features_of_layers[i + 1])
            self.siren_layers.append(layer_i)

    def forward(self, x):
        out = x
        for layer_i in self.siren_layers:
            out = layer_i(out)
        return out


def get_cameraman_tensor(sidelength):
    # img = Image.fromarray(skimage.data.camera())
    img = Image.open('./image.jpeg')
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).reshape(-1, 3)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

img_siren = SIREN([2, 256, 256, 256, 1])
img_siren.cuda()

total_steps = 5000  # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 100

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

for step in range(total_steps):
    model_output = img_siren(model_input)
    loss = ((model_output - ground_truth) ** 2).mean()

    if not step % steps_til_summary:
        model_output = img_siren(model_input)
        print("Step %d, Total loss %0.6f" % (step, loss))
        plt.imshow(model_output.cpu().view(256, 256).detach().numpy())
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()
