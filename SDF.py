"""
In this module I will use a SIREN to approximate the SDF function of a given point cloud.
The SDF function is negative inside the object, positive outside and zero on its boundary.
The point clouds will be taken from ShapeNet dataset.
"""
from torch.optim import Adam
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

lr = 0.0001
steps = 5000
sumary_after_n_steps = 500


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


def get_random_point_cloud_from_shape_net(dataset):
    N = dataset.len()
    rand_idx = np.random.randint(0, N - 1)
    print(rand_idx)
    return dataset[rand_idx]


import matplotlib.pyplot as plt


def plot_points(point_cloud_pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = point_cloud_pos[:, 0], point_cloud_pos[:, 1], point_cloud_pos[:, 2]
    ax.scatter(xs, ys, zs)
    plt.show()


def plot_sdf_test(siren):  # already trained on the point cloud so we don't need it
    with torch.no_grad():
        new_points = torch.rand((3000, 3)) * 200 - 100
        sdf_approx, _ = siren(new_points)
        new_points = new_points[sdf_approx.squeeze(-1) <= 0.]
        plot_points(new_points)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def sdf_training_step(siren, point_cloud):
    """
        The main problem is the "signed". If it wasn't signed the problem was easy because we could sample far points
        and calculate diffrentiable ditance...
        To overcome the signed issue the normals are necessary. We want the gradient on the boundary be in the opposite
        direction of the normal (i.e. positive outside, negative inside).
    """
    boundary = point_cloud.pos  # pos (the points 3D locations)
    boundary_normals = point_cloud.x  # x (features are the normals)
    boundary.requires_grad_()

    # rand_move = torch.rand(boundary.shape[0]) * 2  # half inside half outside
    # off_boundary_points = rand_move.unsqueeze(-1).expand(*boundary.shape) * boundary
    off_boundary_points = torch.rand((10000, 3))
    # calculate SIREN (sdf approx) gradient w.r.t the boundary
    # we can also first SIREN derivative w.r.t the input and then use it (as approx').
    approx_sdf, coords = siren(boundary)
    siren_grad_on_boundary = gradient(approx_sdf, coords)
    siren_grad_on_boundary_norm = torch.sqrt((siren_grad_on_boundary ** 2).mean(dim=-1))

    # no need to divide in the normals norm because is a constant
    normals_loss = (siren_grad_on_boundary * boundary_normals).mean(dim=-1) / siren_grad_on_boundary_norm
    normals_loss = normals_loss.mean()
    on_boundary_loss = approx_sdf.abs().mean()  # on boundary with low sdf
    gradients_loss = ((siren_grad_on_boundary - 1) ** 2).mean()
    off_boundary_points_sdf_approx, _ = siren(off_boundary_points)
    off_boundary_loss = torch.exp(
        -off_boundary_points_sdf_approx.abs()).mean()  # off boundary points to be with higher sdf

    loss = 50* gradients_loss + 100 * normals_loss + 3000 * on_boundary_loss + 3000 * off_boundary_loss
    # loss = 0.3 * on_boundary_loss + 0.3 * off_boundary_loss
    return loss


def sdf_evaluate(siren, point_cloud):
    with torch.no_grad():
        boundary = point_cloud.pos
        rand_move = torch.rand(boundary.shape[0]) * 2
        off_boundary_points = rand_move.unsqueeze(-1).expand(*boundary.shape) * boundary

        approx_sdf, coords = siren(boundary)
        on_boundary_loss = approx_sdf.abs().mean()  # on boundary with low sdf

        off_boundary_points_sdf_approx, _ = siren(off_boundary_points)
        off_boundary_loss = off_boundary_points_sdf_approx.abs().mean()  # off boundary points to be with higher sdf

        print(f"mean abs SDF of boundary points: {on_boundary_loss}")
        print(f"mean abs SDF of off-boundary points: {off_boundary_loss}")


def sdf_train(siren, optimizer, point_cloud):
    for step_num in range(steps):
        loss = sdf_training_step(siren, point_cloud)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_num != 0 and step_num % sumary_after_n_steps == 0:
            print(f"loss of step {step_num}: {loss}")
            sdf_evaluate(siren, point_cloud)
            plot_sdf_test(siren)
            plot_points(point_cloud.pos.detach())


if __name__ == '__main__':
    from torch_geometric import datasets

    _dataset = datasets.ShapeNet(root='./data/shape_net', split='train', include_normals=True)
    _point_cloud = get_random_point_cloud_from_shape_net(_dataset)
    _point_cloud.pos -= _point_cloud.pos.permute(1, 0).mean(-1).unsqueeze(0).expand_as(_point_cloud.pos)
    _point_cloud.pos /= _point_cloud.pos.permute(1, 0).abs().max(-1)[0].unsqueeze(0).expand_as(_point_cloud.pos)
    _point_cloud.pos *= 100

    # _siren = SIREN([3, 256, 256, 1])  # (x, y, z)  -->  signed distance (d)
    _siren = Siren(3, 256, 3, 1)  # (x, y, z)  -->  signed distance (d)
    _optimizer = Adam(_siren.parameters(), lr=lr)

    sdf_train(_siren, _optimizer, _point_cloud)
