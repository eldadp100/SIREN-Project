import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SIRENLayer, self).__init__()
        self.in_features = in_features
        self.lin = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        self.lin.weight.uniform_(-1/self.in_features, 1/self.in_features)

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





# ======== Experiments ==========
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, img):
        self.img = img
        self.rows = img.shape[0]
        self.columns = img.shape[1]

    def __getitem__(self, i):
        row = i // self.columns
        column = i % self.columns

        if row > self.rows:
            raise Exception("not valid index")
        return torch.tensor([row, column]).type(torch.FloatTensor), self.img[:, row, column]

    def __len__(self):
        return self.rows * self.columns


"""
    TODO: optimize this function to be able to parallelize a batch of images adn utilize the GPU.
"""


def train_implicit_image(img, siren, optimizer, loss_fn):
    img_ds = ImageDataset(img)  # (x,y) -> RGB value
    dl = DataLoader(img_ds, batch_size=1, shuffle=True,pin_memory=True)
    epochs = 100
    for _ in range(epochs):
        for batch in dl:
            indices, values = batch
            indices, values = indices.to(device), values.to(device)
            values_preds = siren(indices)

            loss = loss_fn(values_preds, values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


""" create once and then use clone - maybe do as factoring class to do this as default"""


class GridCreate:
    current_grid = None

    @staticmethod
    def create_2d_grid(rows, columns):
        with torch.no_grad():
            if GridCreate.current_grid is None:
                grid = torch.empty((rows, columns, 2))
                for i in range(columns):
                    for j in range(rows):
                        grid[i][j] = torch.tensor([i, j]).type(torch.FloatTensor)
                GridCreate.current_grid = grid

        return GridCreate.current_grid.detach().clone()


from PIL import Image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt

def l2_loss(a, b):
    return torch.sum(torch.square(a - b))


if __name__ == '__main__':
    _lr = 0.0001
    _model = SIREN([2, 100, 200, 100, 3]).to(device)  # (x, y) --> rep layer 1 --> ... -> RGB
    _loss_fn = l2_loss
    _optimizer = torch.optim.Adam(_model.parameters(), lr=_lr)

    # load image
    img = Image.open("./image.jpeg")
    img_to_shape = (1024, 1024)
    img = ToTensor()(Resize(img_to_shape)(img))

    show_image = False
    if show_image:
        plt.imshow(img)
        plt.title("original image")
        plt.show()

    # train siren
    train_implicit_image(img, _model, _optimizer, _loss_fn)

    # test it
    grid = GridCreate.create_2d_grid(*img_to_shape).to(device)
    reshaped_grid = grid.reshape(-1, 2)
    reshaped_grid = _model(reshaped_grid).detach().cpu()
    reshaped_grid = reshaped_grid.view((*img_to_shape, 3))
    plt.imshow(reshaped_grid)
    plt.title("sampled from SIREN")
    plt.show()


