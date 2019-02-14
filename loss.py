import torch
from torch.nn import functional as F


def d_lsgan_loss(discriminator, trues, fakes, alpha):
    d_trues = discriminator.forward(trues, alpha)
    d_fakes = discriminator.forward(fakes, alpha)

    loss = F.mse_loss(d_trues, torch.ones_like(d_trues)) + F.mse_loss(d_fakes, torch.zeros_like(d_fakes))
    loss /= 2
    return loss


def g_lsgan_loss(discriminator, fakes, alpha):
    d_fakes = discriminator.forward(fakes, alpha)
    loss = F.mse_loss(d_fakes, torch.ones_like(d_fakes)) / 2
    return loss


def d_wgan_loss(discriminator, trues, fakes, alpha):
    epsilon_drift = 1e-3
    lambda_gp = 10

    batch_size = fakes.size()[0]
    d_trues = discriminator.forward(trues, alpha)
    d_fakes = discriminator.forward(fakes, alpha)

    loss_wd = (d_trues - d_fakes).mean()

    # gradient penalty
    epsilon = torch.randn(batch_size, 1, 1, 1, dtype=fakes.dtype, device=fakes.device)
    intpl = epsilon * fakes + (1 - epsilon) * trues
    intpl.requires_grad_()
    f = discriminator.forward(intpl, alpha)
    df = torch.autograd.grad(f, intpl,
                             grad_outputs=torch.ones(*f.size(), device=f.device, dtype=f.dtype),
                             retain_graph=True, create_graph=True, only_inputs=True)[0]
    df_norm = df.view(batch_size, -1).norm(dim=1)
    loss_gp = lambda_gp * ((df_norm - 1) ** 2).mean()

    # drift
    loss_drift = epsilon_drift * (d_trues ** 2).mean()

    loss = -loss_wd + loss_gp + loss_drift

    wd = loss_wd.mean().item()

    return loss, wd


def g_wgan_loss(discriminator, fakes, alpha):
    d_fakes = discriminator.forward(fakes, alpha)
    loss = -d_fakes.mean()
    return loss
