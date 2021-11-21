import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
from dataset import MultiScaleDataset, ImageDataset, MultiScalePatchDataset, MultiScalePatchProgressiveDataset, MultiScalePatchProgressivePairedDataset
from calculate_fid import calculate_fid, calculate_fid_ddp
from distributed import get_rank, synchronize, reduce_loss_dict
from tensor_transforms import convert_to_coord_format

from torchvision.transforms import functional as trans_fn

try:
    import nsml
    from nsml import DATASET_PATH, SESSION_NAME
except ImportError:
    nsml = None


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_coordinate_l2_loss(coord_h, pred_coord_h, coord_w, pred_coord_w):
    loss = nn.MSELoss(reduction='sum')
    h_loss = loss(coord_h, pred_coord_h.squeeze())
    w_loss = loss(coord_w, pred_coord_w.squeeze())

    return h_loss.mean() + w_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_structure_l2_loss(fake_img, structure_img, fake_crop_params, structure_crop_params):
    mse = nn.MSELoss(reduction='mean')
    loss = None

    for i in range(fake_img.shape[0]):
        magnification = fake_crop_params[0][i] / structure_crop_params[0][i]
        downsampled_fake_img = trans_fn.resize(fake_img[i], int(fake_img[i].shape[2]/magnification))
        cropped_structure_img = structure_img[i, :, int(fake_crop_params[2][i]/magnification - structure_crop_params[2][i]) : int(fake_crop_params[2][i]/magnification - structure_crop_params[2][i]) + downsampled_fake_img.shape[2], int(fake_crop_params[3][i]/magnification - structure_crop_params[3][i]) : int(fake_crop_params[3][i]/magnification - structure_crop_params[3][i]) + downsampled_fake_img.shape[2]]
        if i == 0:
            loss = mse(downsampled_fake_img, cropped_structure_img)
        else:
            loss += mse(downsampled_fake_img, cropped_structure_img)

    return loss / fake_img.shape[0]


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def train(args, loader, loader2, generator, discriminator, g_optim, d_optim, g_ema, g_ema_temp, device, fid_dataset, fid_dataset2, n_scales, writer, path):
    loader = sample_data(loader)
    loader2 = sample_data(loader2)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
        
    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    ################################################## phase1 training
    g_module.update_pe_mask(unmasking_ratio=args.unmasking_ratio1, device=device)
    g_ema_temp.update_pe_mask(unmasking_ratio=args.unmasking_ratio1, device=device)

    for p1iter in range(args.iter):
        print("phase 1, iteration", p1iter)
        data = next(loader)
        key = np.random.randint(n_scales)
        real_stack = data[key].to(device)

        real_img, converted = real_stack[:, :3], real_stack[:, 3:]

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _ = generator(converted, noise)
        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        fake_pred = discriminator(fake, key)

        real = real_img if args.img2dis else real_stack
        real_pred = discriminator(real, key)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        # loss_dict['d'] += d_loss
        # loss_dict['d'] /= 2
        # loss_dict['d_main'] += d_loss
        # loss_dict['d_main'] /= 2
        # loss_dict['real_score'] += real_pred.mean()
        # loss_dict['real_score'] /= 2
        # loss_dict['fake_score'] += fake_pred.mean()
        # loss_dict['fake_score'] /= 2

        loss_dict['p1image_d'] = d_loss
        loss_dict['p1image_d_main'] = d_loss
        loss_dict['p1image_real_score'] = real_pred.mean()
        loss_dict['p1image_fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = p1iter % args.d_reg_every == 0

        if d_regularize:
            real.requires_grad = True
            real_pred = discriminator(real, key)
            r1_loss = d_r1_loss(real_pred, real)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        # loss_dict['r1'] += r1_loss
        # loss_dict['r1'] /= 2

        loss_dict['p1image_r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _ = generator(converted, noise)
        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        fake_pred = discriminator(fake, key)
        g_loss = g_nonsaturating_loss(fake_pred)

        # loss_dict['g'] += g_loss
        # loss_dict['g'] /= 2
        # loss_dict['g_main'] += g_loss
        # loss_dict['g_main'] /= 2

        loss_dict['p1image_g'] = g_loss
        loss_dict['p1image_g_main'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema_temp, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        imagep1_d_loss_val = loss_reduced['p1image_d'].mean().item()
        imagep1_d_loss_main_val = loss_reduced['p1image_d_main'].mean().item()
        imagep1_g_loss_val = loss_reduced['p1image_g'].mean().item()
        imagep1_g_loss_main_val = loss_reduced['p1image_g_main'].mean().item()
        imagep1_r1_val = loss_reduced['p1image_r1'].mean().item()
        imagep1_real_score_val = loss_reduced['p1image_real_score'].mean().item()
        imagep1_fake_score_val = loss_reduced['p1image_fake_score'].mean().item()

        if p1iter % 100 == 0:
            if nsml:
                nsml.report(summary=True, step=p1iter,
                # Generator=g_loss_val,
                # G_main=g_loss_main_val,
                # G_aux=g_loss_aux_val,
                # Discriminator=d_loss_val,
                # D_main=d_loss_main_val,
                # D_aux=d_loss_aux_val,
                # R1=r1_val,
                # PathLengthRegularization=path_loss_val,
                # MeanPathLength=mean_path_length,
                # RealScore=real_score_val,
                # FakeScore=fake_score_val,
                # PathLength=path_length_val,
                # Patch1x_Generator=patch1x_g_loss_val,
                # Patch1x_G_main=patch1x_g_loss_main_val,
                # Patch1x_G_aux=patch1x_g_loss_aux_val,
                # Patch1x_Discriminator=patch1x_d_loss_val,
                # Patch1x_D_main=patch1x_d_loss_main_val,
                # Patch1x_D_aux=patch1x_d_loss_aux_val,
                # Patch1x_R1=patch1x_r1_val,
                # Patch1x_RealScore=patch1x_real_score_val,
                # Patch1x_FakeScore=patch1x_fake_score_val,
                P1_Generator=imagep1_g_loss_val,
                P1_G_main=imagep1_g_loss_main_val,
                P1_Discriminator=imagep1_d_loss_val,
                P1_D_main=imagep1_d_loss_main_val,
                P1_R1=imagep1_r1_val,
                P1_RealScore=imagep1_real_score_val,
                P1_FakeScore=imagep1_fake_score_val)
            # else:
                # writer.add_scalar("Generator", g_loss_val, i)
                # writer.add_scalar("Discriminator", d_loss_val, i)
                # writer.add_scalar("R1", r1_val, i)
                # writer.add_scalar("Path Length Regularization", path_loss_val, i)
                # writer.add_scalar("Mean Path Length", mean_path_length, i)
                # writer.add_scalar("Real Score", real_score_val, i)
                # writer.add_scalar("Fake Score", fake_score_val, i)
                # writer.add_scalar("Path Length", path_length_val, i)
        if p1iter % 500 == 0:
            with torch.no_grad():
                g_ema_temp.eval()
                converted_full = convert_to_coord_format(sample_z.size(0), int(args.size/4), int(args.size/4), device,
                                                            integer_values=args.coords_integer_values)
                if args.generate_by_one:
                    converted_full = convert_to_coord_format(1, int(args.size/4), int(args.size/4), device,
                                                                integer_values=args.coords_integer_values)
                    samples = []
                    for sz in sample_z:
                        sample, _ = g_ema_temp(converted_full, [sz.unsqueeze(0)])
                        samples.append(sample)
                    sample = torch.cat(samples, 0)
                else:
                    sample, _ = g_ema_temp(converted_full, [sample_z])

                utils.save_image(
                    sample,
                    os.path.join(path, 'outputs', args.output_dir, 'p1_images', f'{str(p1iter).zfill(6)}.png'),
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )

                if p1iter == 0:
                    utils.save_image(
                        fake_img,
                        os.path.join(
                            path,
                            f'outputs/{args.output_dir}/p1_images/fake_patch_{str(key)}_{str(p1iter).zfill(6)}.png'),
                        nrow=int(fake_img.size(0) ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        real_img,
                        os.path.join(
                            path,
                            f'outputs/{args.output_dir}/p1_images/real_patch_{str(key)}_{str(p1iter).zfill(6)}.png'),
                        nrow=int(real_img.size(0) ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        if p1iter % args.save_checkpoint_frequency == 0:
            if p1iter > 0:
                cur_metrics = calculate_fid_ddp(g_ema_temp, fid_dataset=fid_dataset, bs=args.fid_batch, size=int(args.coords_size/args.patch_multiplier),
                                            num_batches=args.fid_samples//args.fid_batch, latent_size=args.latent,
                                            save_dir=args.path_fid, integer_values=args.coords_integer_values)
                if nsml:
                    nsml.report(summary=True, step=p1iter, p1_fid=cur_metrics['frechet_inception_distance'])
                else:
                    writer.add_scalar("p1_fid", cur_metrics['frechet_inception_distance'], p1iter)
                print(p1iter, "p1_fid",  cur_metrics['frechet_inception_distance'])
    ################################################## phase1 training

    # g_module.update_pe_mask(unmasking_ratio=args.unmasking_ratio2, device=device)
    g_module.progressive_open_alpha = 0
    g_ema.update_pe_mask(unmasking_ratio=args.unmasking_ratio2, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        g_module.progressive_open_pe_mask(unmasking_ratio=args.unmasking_ratio2, previous_unmasking_ratio=args.unmasking_ratio1, total_iteration=args.pgopen_iter, curent_iteration=i, device=device)

        if i > args.iter:
            print('Done!')

            break

        data, coord_h, coord_w, fake_crop_params, structure_crop_params = next(loader2)
        key = np.random.randint(n_scales)
        real_stack = data[key].to(device)

        real_img, converted, structured = real_stack[:, :3], real_stack[:, 3:5], real_stack[:, 5:7]

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _ = generator(converted, noise)
        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        fake_pred = discriminator(fake, key)

        real = real_img if args.img2dis else real_stack
        real_pred = discriminator(real, key)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real.requires_grad = True
            real_pred = discriminator(real, key)
            r1_loss = d_r1_loss(real_pred, real)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _ = generator(converted, noise)
        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        g_ema_temp.eval()
        structure_img, _ = g_ema_temp(structured, noise)
        fake_pred = discriminator(fake, key)
        g_loss_structure = args.structure_loss*g_structure_l2_loss(fake_img, structure_img, fake_crop_params, structure_crop_params)
        g_loss_main = g_nonsaturating_loss(fake_pred)
        g_loss = g_loss_structure + g_loss_main

        loss_dict['g'] = g_loss
        loss_dict['g_main'] = g_loss_main
        loss_dict['g_structure'] = g_loss_structure

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        g_loss_main_val = loss_reduced['g_main'].mean().item()
        g_loss_structure_val = loss_reduced['g_structure'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                    f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'
                )
            )

            if i % 100 == 0:
                if nsml:
                    nsml.report(summary=True, step=i,
                    Generator=g_loss_val,
                    GeneratorMain=g_loss_main_val,
                    GeneratorStructure=g_loss_structure_val,
                    Discriminator=d_loss_val,
                    R1=r1_val,
                    PathLengthRegularization=path_loss_val,
                    MeanPathLength=mean_path_length,
                    RealScore=real_score_val,
                    FakeScore=fake_score_val,
                    PathLength=path_length_val)
                else:
                    writer.add_scalar("Generator", g_loss_val, i)
                    writer.add_scalar("Discriminator", d_loss_val, i)
                    writer.add_scalar("R1", r1_val, i)
                    writer.add_scalar("Path Length Regularization", path_loss_val, i)
                    writer.add_scalar("Mean Path Length", mean_path_length, i)
                    writer.add_scalar("Real Score", real_score_val, i)
                    writer.add_scalar("Fake Score", fake_score_val, i)
                    writer.add_scalar("Path Length", path_length_val, i)

            if i % 500 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    converted_full = convert_to_coord_format(sample_z.size(0), args.size, args.size, device,
                                                             integer_values=args.coords_integer_values)
                    if args.generate_by_one:
                        converted_full = convert_to_coord_format(1, args.size, args.size, device,
                                                                 integer_values=args.coords_integer_values)
                        samples = []
                        for sz in sample_z:
                            sample, _ = g_ema(converted_full, [sz.unsqueeze(0)])
                            samples.append(sample)
                        sample = torch.cat(samples, 0)
                    else:
                        sample, _ = g_ema(converted_full, [sample_z])

                    utils.save_image(
                        sample,
                        os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}.png'),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    if i == 0:
                        utils.save_image(
                            fake_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images/fake_patch_{str(key)}_{str(i).zfill(6)}.png'),
                            nrow=int(fake_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            real_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}.png'),
                            nrow=int(real_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
            if i % 500 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    converted_full = convert_to_coord_format(sample_z.size(0), int(args.size/4), int(args.size/4), device,
                                                             integer_values=args.coords_integer_values)
                    if args.generate_by_one:
                        converted_full = convert_to_coord_format(1, int(args.size/4), int(args.size/4), device,
                                                                 integer_values=args.coords_integer_values)
                        samples = []
                        for sz in sample_z:
                            sample, _ = g_ema(converted_full, [sz.unsqueeze(0)])
                            samples.append(sample)
                        sample = torch.cat(samples, 0)
                    else:
                        sample, _ = g_ema(converted_full, [sample_z])

                    utils.save_image(
                        sample,
                        os.path.join(path, 'outputs', args.output_dir, 'images_4x_mini', f'{str(i).zfill(6)}.png'),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    if i == 0:
                        utils.save_image(
                            fake_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images_4x_mini/fake_patch_{str(key)}_{str(i).zfill(6)}.png'),
                            nrow=int(fake_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            real_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images_4x_mini/real_patch_{str(key)}_{str(i).zfill(6)}.png'),
                            nrow=int(real_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

            if i % args.save_checkpoint_frequency == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_ema_temp': g_ema_temp.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    os.path.join(
                        path,
                        f'outputs/{args.output_dir}/checkpoints/{str(i).zfill(6)}.pt'),
                )
                if i > 0:
                    cur_metrics = calculate_fid_ddp(g_ema, fid_dataset=fid_dataset2, bs=args.fid_batch, size=args.coords_size,
                                                num_batches=args.fid_samples//args.fid_batch, latent_size=args.latent,
                                                save_dir=args.path_fid, integer_values=args.coords_integer_values)
                    if nsml:
                        nsml.report(summary=True, step=i, fid=cur_metrics['frechet_inception_distance'])
                    else:
                        writer.add_scalar("fid", cur_metrics['frechet_inception_distance'], i)
                    print(i, "fid",  cur_metrics['frechet_inception_distance'])


def ddp_worker(rank, world_size, args):
    path = args.out_path

    Generator = getattr(model, args.Generator)
    print('Generator', Generator)
    Discriminator = getattr(model, args.Discriminator)
    print('Discriminator', Discriminator)
    
    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'images_4x_mini'), exist_ok=True)
    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'p1_images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'checkpoints'), exist_ok=True)
    args.logdir = os.path.join(path, 'tensorboard', args.output_dir)
    os.makedirs(args.logdir, exist_ok=True)
    args.path_fid = os.path.join(path, 'fid', args.output_dir)
    os.makedirs(args.path_fid, exist_ok=True)

    # n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(rank)
        device = torch.device("cuda:{}".format(rank))
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:32797', rank=rank, world_size=world_size)
        synchronize()

    # args.n_mlp = 8
    args.dis_input_size = 3 if args.img2dis else 5
    print('img2dis', args.img2dis, 'dis_input_size', args.dis_input_size)

    args.start_iter = 0
    n_scales = int(math.log(args.size//args.crop, 2)) + 1
    print('n_scales', n_scales)

    generator = Generator(size=int(args.size/args.patch_multiplier), hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                        activation=args.activation, channel_multiplier=args.channel_multiplier,
                        ).to(device)

    print('generator N params', sum(p.numel() for p in generator.parameters() if p.requires_grad))
    discriminator = Discriminator(
        size=int(args.crop/args.patch_multiplier), channel_multiplier=args.channel_multiplier, n_scales=n_scales, input_size=args.dis_input_size,
        n_first_layers=args.n_first_layers,
    ).to(device)
    g_ema = Generator(size=int(args.size/args.patch_multiplier), hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                    activation=args.activation, channel_multiplier=args.channel_multiplier,
                    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    g_ema_temp = Generator(size=int(args.size/args.patch_multiplier), hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                    activation=args.activation, channel_multiplier=args.channel_multiplier,
                    ).to(device)
    g_ema_temp.eval()
    accumulate(g_ema_temp, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)

        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])
        g_ema_temp.load_state_dict(ckpt['g_ema_temp'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

        del ckpt
        torch.cuda.empty_cache()

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform_fid = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul_(255.).byte())])

    if nsml:
        args.path = os.path.join(DATASET_PATH, 'train', args.path.split('/')[-1])
        args.path2 = os.path.join(DATASET_PATH, 'train', args.path2.split('/')[-1])
    else:
        pass
    dataset = MultiScaleDataset(args.path, transform=transform, resolution=int(args.coords_size/args.patch_multiplier), crop_size=int(args.crop/args.patch_multiplier),
                                integer_values=args.coords_integer_values, to_crop=args.to_crop)
    dataset2 = MultiScalePatchProgressivePairedDataset(args.path2, transform=transform, resolution=args.coords_size, resolution_bpg=int(args.coords_size/args.patch_multiplier), crop_size=int(args.crop/args.patch_multiplier),
                                integer_values=args.coords_integer_values, to_crop=args.to_crop)
    fid_dataset = ImageDataset(args.path, transform=transform_fid, resolution=int(args.coords_size/args.patch_multiplier), to_crop=args.to_crop)
    fid_dataset.length = args.fid_samples
    fid_dataset2 = ImageDataset(args.path2, transform=transform_fid, resolution=args.coords_size, to_crop=args.to_crop)
    fid_dataset2.length = args.fid_samples
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    loader2 = data.DataLoader(
        dataset2,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    writer = SummaryWriter(log_dir=args.logdir)

    train(args, loader, loader2, generator, discriminator, g_optim, d_optim, g_ema, g_ema_temp, device, fid_dataset, fid_dataset2, n_scales, writer, path)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str)
    parser.add_argument('path2', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--out_path', type=str, default='.')
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--world_size', type=int, default=2)

    # fid
    parser.add_argument('--fid_samples', type=int, default=50000)
    parser.add_argument('--fid_batch', type=int, default=16)

    # training
    parser.add_argument('--iter', type=int, default=1200000)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--generate_by_one', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--save_checkpoint_frequency', type=int, default=20000)
    parser.add_argument('--structure_loss', type=float, default=0.1)
    parser.add_argument('--pgopen_iter', type=int, default=2000)

    # dataset
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--to_crop', action='store_true')
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--coords_size', type=int, default=256)
    parser.add_argument('--patch_multiplier', type=int, default=4)

    # Generator params
    parser.add_argument('--Generator', type=str, default='ModSIREN')
    parser.add_argument('--coords_integer_values', action='store_true')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--mixing', type=float, default=0.)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--unmasking_ratio1', type=float, default=0.0)
    parser.add_argument('--unmasking_ratio2', type=float, default=0.0)

    # Discriminator params
    parser.add_argument('--Discriminator', type=str, default='Discriminator')
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--img2dis',  action='store_true')
    parser.add_argument('--n_first_layers', type=int, default=0)

    args = parser.parse_args()

    if not args.distributed:
        path = args.out_path

        Generator = getattr(model, args.Generator)
        print('Generator', Generator)
        Discriminator = getattr(model, args.Discriminator)
        print('Discriminator', Discriminator)

        os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'images_4x_mini'), exist_ok=True)
        os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'p1_images'), exist_ok=True)
        os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'checkpoints'), exist_ok=True)
        args.logdir = os.path.join(path, 'tensorboard', args.output_dir)
        os.makedirs(args.logdir, exist_ok=True)
        args.path_fid = os.path.join(path, 'fid', args.output_dir)
        os.makedirs(args.path_fid, exist_ok=True)

        # n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        # args.distributed = n_gpu > 1

        # if args.distributed:
        #     torch.cuda.set_device(args.local_rank)
        #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
        #     synchronize()

        # args.n_mlp = 8
        args.dis_input_size = 3 if args.img2dis else 5
        print('img2dis', args.img2dis, 'dis_input_size', args.dis_input_size)

        args.start_iter = 0
        n_scales = int(math.log(args.size//args.crop, 2)) + 1
        print('n_scales', n_scales)

        generator = Generator(size=int(args.size/args.patch_multiplier), hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                            activation=args.activation, channel_multiplier=args.channel_multiplier,
                            ).to(device)

        print('generator N params', sum(p.numel() for p in generator.parameters() if p.requires_grad))
        discriminator = Discriminator(
            size=int(args.crop/args.patch_multiplier), channel_multiplier=args.channel_multiplier, n_scales=n_scales, input_size=args.dis_input_size,
            n_first_layers=args.n_first_layers,
        ).to(device)
        g_ema = Generator(size=int(args.size/args.patch_multiplier), hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                        activation=args.activation, channel_multiplier=args.channel_multiplier,
                        ).to(device)
        g_ema.eval()
        accumulate(g_ema, generator, 0)
        g_ema_temp = Generator(size=int(args.size/args.patch_multiplier), hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                        activation=args.activation, channel_multiplier=args.channel_multiplier,
                        ).to(device)
        g_ema_temp.eval()
        accumulate(g_ema_temp, generator, 0)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        if args.ckpt is not None:
            print('load model:', args.ckpt)

            ckpt = torch.load(args.ckpt)

            try:
                ckpt_name = os.path.basename(args.ckpt)
                args.start_iter = int(os.path.splitext(ckpt_name)[0])

            except ValueError:
                pass

            generator.load_state_dict(ckpt['g'])
            discriminator.load_state_dict(ckpt['d'])
            g_ema.load_state_dict(ckpt['g_ema'])
            g_ema_temp.load_state_dict(ckpt['g_ema_temp'])

            g_optim.load_state_dict(ckpt['g_optim'])
            d_optim.load_state_dict(ckpt['d_optim'])

            del ckpt
            torch.cuda.empty_cache()

        # if args.distributed:
        #     generator = nn.parallel.DistributedDataParallel(
        #         generator,
        #         device_ids=[args.local_rank],
        #         output_device=args.local_rank,
        #         broadcast_buffers=False,
        #     )

        #     discriminator = nn.parallel.DistributedDataParallel(
        #         discriminator,
        #         device_ids=[args.local_rank],
        #         output_device=args.local_rank,
        #         broadcast_buffers=False,
        #     )

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform_fid = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.mul_(255.).byte())])

        if nsml:
            args.path = os.path.join(DATASET_PATH, 'train', args.path.split('/')[-1])
            args.path2 = os.path.join(DATASET_PATH, 'train', args.path2.split('/')[-1])
        else:
            pass
        dataset = MultiScaleDataset(args.path, transform=transform, resolution=int(args.coords_size/args.patch_multiplier), crop_size=int(args.crop/args.patch_multiplier),
                                    integer_values=args.coords_integer_values, to_crop=args.to_crop)
        dataset2 = MultiScalePatchProgressivePairedDataset(args.path2, transform=transform, resolution=args.coords_size, resolution_bpg=int(args.coords_size/args.patch_multiplier), crop_size=int(args.crop/args.patch_multiplier),
                                    integer_values=args.coords_integer_values, to_crop=args.to_crop)
        fid_dataset = ImageDataset(args.path, transform=transform_fid, resolution=int(args.coords_size/args.patch_multiplier), to_crop=args.to_crop)
        fid_dataset.length = args.fid_samples
        fid_dataset2 = ImageDataset(args.path2, transform=transform_fid, resolution=args.coords_size, to_crop=args.to_crop)
        fid_dataset2.length = args.fid_samples
        loader = data.DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        loader2 = data.DataLoader(
            dataset2,
            batch_size=args.batch,
            sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        writer = SummaryWriter(log_dir=args.logdir)

        train(args, loader, loader2, generator, discriminator, g_optim, d_optim, g_ema, g_ema_temp, device, fid_dataset, fid_dataset2, n_scales, writer, path)
    else:
        world_size = args.world_size
        mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)