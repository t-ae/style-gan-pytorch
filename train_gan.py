#!/usr/bin/env python

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
import torch
import torch.nn.functional as F
import torch.backends.cudnn
import torch.autograd
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
import torchvision

import network
import data_loader
import image_converter
import utils

# parameters
SETTING_JSON_PATH = "./settings.json"
EPSILON_DRIFT = 1e-3
LAMBDA_GP = 10


def main():
    # load settings
    with open(SETTING_JSON_PATH) as fp:
        settings = json.load(fp)
    output_root = Path("./output").joinpath(datetime.now(timezone(timedelta(hours=+9), 'JST')).strftime("%Y%m%d_%H%M%S"))
    output_root.mkdir()
    shutil.copy(SETTING_JSON_PATH, output_root.joinpath("settings.json"))

    if settings["detect_anomaly"]:
        with torch.autograd.detect_anomaly():
            train(settings, output_root)
    else:
        train(settings, output_root)


def d_wgan_loss(discriminator, trues, fakes, alpha):
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
    loss_gp = LAMBDA_GP * ((df_norm - 1) ** 2).mean()

    # drift
    loss_drift = EPSILON_DRIFT * (d_trues ** 2).mean()

    loss = -loss_wd + loss_gp + loss_drift

    wd = loss_wd.mean().item()

    return loss, wd


def d_lsgan_loss(discriminator, trues, fakes, alpha):
    d_trues = discriminator.forward(trues, alpha)
    d_fakes = discriminator.forward(fakes, alpha)

    loss = F.mse_loss(d_trues, torch.ones_like(d_trues)) + F.mse_loss(d_fakes, torch.zeros_like(d_fakes))
    loss /= 2
    return loss


def g_wgan_loss(discriminator, fakes, alpha):
    d_fakes = discriminator.forward(fakes, alpha)
    loss = -d_fakes.mean()
    return loss


def g_lsgan_loss(discriminator, fakes, alpha):
    d_fakes = discriminator.forward(fakes, alpha)
    loss = F.mse_loss(d_fakes, torch.ones_like(d_fakes)) / 2
    return loss


def train(settings, output_root):
    # directories
    weights_root = output_root.joinpath("weights")
    weights_root.mkdir()

    # settings
    if settings["use_cuda"]:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dtype = torch.float32
    test_device = torch.device("cuda:0")
    test_dtype = torch.float16

    loss_type = settings["loss"]

    z_dim = settings["network"]["z_dimension"]

    # model
    generator = network.Generator(settings["network"]).to(device, dtype)
    discriminator = network.Discriminator(settings["network"]).to(device, dtype)

    lt_learning_rate = settings["learning_rates"]["latent_transformation"]
    g_learning_rate = settings["learning_rates"]["generator"]
    d_learning_rate = settings["learning_rates"]["discriminator"]
    g_opt = optim.Adam([
        {"params": generator.latent_transform.parameters(), "lr": lt_learning_rate},
        {"params": generator.synthesis_module.parameters()}
    ], lr=g_learning_rate, betas=(0.0, 0.99), eps=1e-8)
    d_opt = optim.Adam(discriminator.parameters(),
                       lr=d_learning_rate, betas=(0.0, 0.99), eps=1e-8)

    # train data
    image_root: Path = Path(__file__).parent.joinpath("../images/")
    image_paths = list(image_root.glob(settings["file_name_pattern"]))

    print(settings["file_name_pattern"])
    print(f"{len(image_paths)} images")
    loader = data_loader.TrainDataLoader(image_paths,
                                         settings["data_augmentation"])
    if settings["use_yuv"]:
        converter = image_converter.YUVConverter()
    else:
        converter = image_converter.RGBConverter()

    # parameters
    level = settings["start_level"]
    generator.set_level(level)
    discriminator.set_level(level)
    fading = False
    alpha = 1
    step = 0

    # log
    writer = SummaryWriter(str(output_root.joinpath("logdir")))
    test_zs = utils.create_test_z(z_dim)
    test_z0 = torch.from_numpy(test_zs[0]).to(test_device, test_dtype)
    test_z1 = torch.from_numpy(test_zs[1]).to(test_device, test_dtype)

    for loop in range(9999999):
        size = 2 ** (level+1)

        batch_size = settings["batch_sizes"][level-1]
        alpha_delta = batch_size / settings["num_images_in_stage"]

        image_count = 0

        for batch in loader.generate(batch_size, size, size):
            # pre train
            step += 1
            image_count += batch_size
            if fading:
                alpha = min(1.0, alpha + alpha_delta)

            # data
            batch = batch.transpose([0, 3, 1, 2])
            batch = converter.to_train_data(batch)
            trues = torch.from_numpy(batch).to(device, dtype)

            # reset
            g_opt.zero_grad()
            d_opt.zero_grad()

            # sample fakes
            z = utils.create_z(batch_size, z_dim)
            z = torch.from_numpy(z).to(device, dtype)
            fakes = generator.forward(z, alpha)
            fakes_nograd = fakes.detach()

            # === train discriminator ===
            if loss_type == "wgan":
                d_loss, wd = d_wgan_loss(discriminator, trues, fakes_nograd, alpha)
            elif loss_type == "lsgan":
                d_loss = d_lsgan_loss(discriminator, trues, fakes_nograd, alpha)
            else:
                raise Exception(f"Invalid loss: {loss_type}")

            d_loss.backward()
            d_opt.step()

            # === train generator ===
            if loss_type == "wgan":
                g_loss = g_wgan_loss(discriminator, fakes, alpha)
            elif loss_type == "lsgan":
                g_loss = g_lsgan_loss(discriminator, fakes, alpha)
            else:
                raise Exception(f"Invalid loss: {loss_type}")

            g_loss.backward()
            g_opt.step()

            # log
            if step % 1 == 0:
                print(f"lv{level}-{step}: "
                      f"a: {alpha:.5f} "
                      f"g: {g_loss.item():.7f} "
                      f"d: {d_loss.item():.7f} ")

                writer.add_scalar(f"lv{level}/loss_gen", g_loss.item(), global_step=step)
                writer.add_scalar(f"lv{level}/loss_disc", d_loss.item(), global_step=step)
                if loss_type == "wgan":
                    writer.add_scalar(f"lv{level}/wd", wd, global_step=step)

            # histogram
            if settings["save_steps"]["histogram"] > 0 and step % settings["save_steps"]["histogram"] == 0:
                for name, param in generator.named_parameters():
                    writer.add_histogram(f"gen/{name}", param.cpu().data.numpy(), step)
                for name, param in discriminator.named_parameters():
                    writer.add_histogram(f"disc/{name}", param.cpu().data.numpy(), step)

            # image
            if step % settings["save_steps"]["image"] == 0:
                fading_text = "fading" if fading else "stabilizing"
                with torch.no_grad():
                    eval_gen = network.Generator(settings["network"]).to(test_device, test_dtype).eval()
                    eval_gen.load_state_dict(generator.state_dict())
                    fakes = eval_gen.forward(test_z0, alpha)
                    fakes = torchvision.utils.make_grid(fakes, nrow=4)
                    fakes = fakes.to(torch.float32).cpu().numpy()
                    fakes = converter.from_generator_output(fakes)
                    writer.add_image(f"lv{level}_{fading_text}/intpl", torch.from_numpy(fakes), step)
                    fakes = eval_gen.forward(test_z1, alpha)
                    fakes = torchvision.utils.make_grid(fakes, nrow=4)
                    fakes = fakes.to(torch.float32).cpu().numpy()
                    fakes = converter.from_generator_output(fakes)
                    writer.add_image(f"lv{level}_{fading_text}/random", torch.from_numpy(fakes), step)

            # model save
            if step % settings["save_steps"]["model"] == 0 and level >= 3 and not fading:
                savedir = weights_root.joinpath(f"{step}_lv{level}")
                savedir.mkdir()
                torch.save(generator.state_dict(), savedir.joinpath("gen.pth"))
                torch.save(discriminator.state_dict(), savedir.joinpath("disc.pth"))

            # fading/stabilizing
            if image_count > settings["num_images_in_stage"]:
                if fading:
                    print("start stabilizing")
                    fading = False
                    alpha = 1
                    image_count = 0
                elif level < settings["max_level"]:
                    print(f"end lv: {level}")
                    break

        # level up
        if level < settings["max_level"]:
            level = level+1
            generator.set_level(level)
            discriminator.set_level(level)
            fading = True
            alpha = 0
            print(f"lv up: {level}")

            if settings["reset_optimizer"]:
                g_opt = optim.Adam([
                    {"params": generator.latent_transform.parameters(), "lr": lt_learning_rate},
                    {"params": generator.synthesis_module.parameters()}
                ], lr=g_learning_rate, betas=(0.0, 0.99), eps=1e-8)
                d_opt = optim.Adam(discriminator.parameters(),
                                   lr=d_learning_rate, betas=(0.0, 0.99), eps=1e-8)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
