import torch
from torch.nn import functional as F


def d_lsgan_loss(discriminator, trues, fakes, labels, alpha):
    d_trues = discriminator.forward(trues, labels, alpha)
    d_fakes = discriminator.forward(fakes, labels, alpha)

    loss = F.mse_loss(d_trues, torch.ones_like(d_trues)) + F.mse_loss(d_fakes, torch.zeros_like(d_fakes))
    loss /= 2
    return loss


def g_lsgan_loss(discriminator, fakes, labels, alpha):
    d_fakes = discriminator.forward(fakes, labels, alpha)
    loss = F.mse_loss(d_fakes, torch.ones_like(d_fakes)) / 2
    return loss


def d_wgan_loss(discriminator, trues, fakes, labels, alpha):
    epsilon_drift = 1e-3
    lambda_gp = 10

    batch_size = fakes.size()[0]
    d_trues = discriminator.forward(trues, labels, alpha)
    d_fakes = discriminator.forward(fakes, labels, alpha)

    loss_wd = d_trues.mean() - d_fakes.mean()

    # gradient penalty
    epsilon = torch.rand(batch_size, 1, 1, 1, dtype=fakes.dtype, device=fakes.device)
    intpl = epsilon * fakes + (1 - epsilon) * trues
    intpl.requires_grad_()
    f = discriminator.forward(intpl, labels, alpha)
    grad = torch.autograd.grad(f.sum(), intpl, create_graph=True)[0]
    grad_norm = grad.view(batch_size, -1).norm(dim=1)
    loss_gp = lambda_gp * ((grad_norm - 1) ** 2).mean()

    # drift
    loss_drift = epsilon_drift * (d_trues ** 2).mean()

    loss = -loss_wd + loss_gp + loss_drift

    wd = loss_wd.item()

    return loss, wd


def g_wgan_loss(discriminator, fakes, labels, alpha):
    d_fakes = discriminator.forward(fakes, labels, alpha)
    loss = -d_fakes.mean()
    return loss


def d_logistic_loss(discriminator, trues, fakes, labels, alpha, r1gamma=10):
    d_fakes = discriminator.forward(fakes, labels, alpha)
    trues.requires_grad_()
    d_trues = discriminator.forward(trues, labels, alpha)
    loss = F.softplus(d_fakes).mean() + F.softplus(-d_trues).mean()

    if r1gamma > 0:
        grad = torch.autograd.grad(d_trues.sum(), trues, create_graph=True)[0]
        loss += r1gamma/2 * (grad**2).sum(dim=(1, 2, 3)).mean()

    return loss


def g_logistic_loss(discriminator, fakes, labels, alpha):
    d_fakes = discriminator.forward(fakes, labels, alpha)
    return F.softplus(-d_fakes).mean()
