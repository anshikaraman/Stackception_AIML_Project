
"""Test a model and generate submission CSV.

> python3 train.py --conf ../cfg/s1.yml

Usage:
    > python train.py --load_path PATH --name NAME
    where
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the train run
"""
import args
# import config
import dataset
import engine
import layers
import util

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torchfile


from PIL import Image
from sklearn import metrics
from sklearn import model_selection
from torch.utils.tensorboard import summary
from torch.utils.tensorboard import FileWriter
from torch.autograd import Variable

print("__"*80)
print("Imports Done...")


def load_stage1(args):
    #* Init models and weights:
    from layers import Stage1Generator, Stage1Discriminator
    netG = Stage1Generator(emb_dim=1024)
    netD = Stage1Discriminator(emb_dim=1024)

    netG.apply(engine.weights_init)
    netD.apply(engine.weights_init)

    #* Load saved model:
    if args.NET_G_path != "":
        netG.load_state_dict(torch.load(args.NET_G_path, map_location=torch.device('cpu')))
        print("__"*80)
        print("Generator loaded from: ", args.NET_G_path)
        print("__"*80)
    if args.NET_D_path != "":
        netD.load_state_dict(torch.load(args.NET_D_path, map_location=torch.device('cpu')))
        print("__"*80)
        print("Discriminator loaded from: ", args.NET_D_path)
        print("__"*80)

    #* Load on device:
    if args.device == "cuda":
        netG.cuda()
        netD.cuda()

    print("__"*80)
    print("GENERATOR:")
    print(netG)
    print("__"*80)
    print("DISCRIMINATOR:")
    print(netD)
    print("__"*80)

    return netG, netD


def load_stage2(args):
    #* Init models and weights:
    from layers import Stage2Generator, Stage2Discriminator, Stage1Generator
    Stage1_G = Stage1Generator(emb_dim=1024)
    netG = Stage2Generator(Stage1_G, emb_dim=1024)
    netD = Stage2Discriminator(emb_dim=1024)
    netG.apply(engine.weights_init)
    netD.apply(engine.weights_init)

    #* Load saved model:
    if args.NET_G_path != "":
        netG.load_state_dict(torch.load(args.NET_G_path, map_location=torch.device('cpu')))
        print("Generator loaded from: ", args.NET_G_path)
    elif args.STAGE1_G_path != "":
        netG.stage1_gen.load_state_dict(torch.load(args.STAGE1_G_path, map_location=torch.device('cpu')))
        print("Generator 1 loaded from: ", args.STAGE1_G_path)
    else:
        print("Please give the Stage 1 generator path")
        return
    
    if args.NET_D_path != "":
        netD.load_state_dict(torch.load(args.NET_D_path, map_location=torch.device('cpu')))
        print("Discriminator loaded from: ", args.NET_D_path)

    #* Load on device:
    if args.device == "cuda":
        netG.cuda()
        netD.cuda()

    print("__"*80)
    print(netG)
    print("__"*80)
    print(netD)
    print("__"*80)

    return netG, netD


def run(args):
    if args.STAGE == 1:
        netG, netD = load_stage1(args)
    else:
        netG, netD = load_stage2(args)

    # Setting up device
    device = torch.device(args.device)

    # Load model
    netG.to(device)
    netD.to(device)

    nz = args.n_z
    batch_size = args.train_bs
    noise = Variable(torch.FloatTensor(batch_size, nz)).to(device)
    with torch.no_grad():
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1)).to(device) # volatile=True
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).to(device)
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).to(device)

    gen_lr = args.TRAIN_GEN_LR
    disc_lr = args.TRAIN_DISC_LR

    lr_decay_step = args.TRAIN_LR_DECAY_EPOCH

    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.TRAIN_DISC_LR, betas=(0.5, 0.999))

    netG_para = []
    for p in netG.parameters():
        if p.requires_grad:
            netG_para.append(p)
    optimizerG = torch.optim.Adam(netG_para, lr=args.TRAIN_GEN_LR, betas=(0.5, 0.999))

    count = 0

    training_set = dataset.CUBDataset(pickl_file=args.train_filenames, img_dir=args.images_dir, cnn_emb=args.cnn_annotations_emb_train, stage=args.STAGE)
    testing_set = dataset.CUBDataset(pickl_file=args.test_filenames, img_dir=args.images_dir, cnn_emb=args.cnn_annotations_emb_test, stage=args.STAGE)
    train_data_loader = torch.utils.data.DataLoader(training_set, batch_size=args.train_bs, num_workers=args.train_workers)
    test_data_loader = torch.utils.data.DataLoader(testing_set, batch_size=args.test_bs, num_workers=args.test_workers)
    # util.check_dataset(training_set)
    # util.check_dataset(testing_set)


    # best_accuracy = 0

    util.make_dir(args.image_save_dir)
    util.make_dir(args.model_dir)
    util.make_dir(args.log_dir)
    summary_writer = FileWriter(args.log_dir)

    for epoch in range(1, args.TRAIN_MAX_EPOCH+1):
        print("__"*80)
        start_t = time.time()

        if epoch % lr_decay_step == 0 and epoch > 0:
            gen_lr *= 0.5
            for param_group in optimizerG.param_groups:
                param_group["lr"] = gen_lr
            disc_lr *= 0.5
            for param_group in optimizerD.param_groups:
                param_group["lr"] = disc_lr
        
        errD, errD_real, errD_wrong, errD_fake, errG, kl_loss, count = engine.train_new_fn(
            train_data_loader, args, netG, netD, real_labels, fake_labels,
            noise, fixed_noise,  optimizerD, optimizerG, epoch, count, summary_writer)
        
        end_t = time.time()
        
        print(f"[{epoch}/{args.TRAIN_MAX_EPOCH}] Loss_D: {errD:.4f}, Loss_G: {errG:.4f}, Loss_KL: {kl_loss:.4f}, Loss_real: {errD_real:.4f}, Loss_wrong: {errD_wrong:.4f}, Loss_fake: {errD_fake:.4f}, Total Time: {end_t-start_t :.2f} sec")
        if epoch % args.TRAIN_SNAPSHOT_INTERVAL == 0 or epoch == 1:
            util.save_model(netG, netD, epoch, args)
    
    util.save_model(netG, netD, args.TRAIN_MAX_EPOCH, args)
    summary_writer.close()


def sample(args, data_loader, noise, fixed_noise, epoch):
    if args.STAGE == 1:
        netG, _ = load_stage1(args)
    else:
        netG, _ = load_stage2(args)
    netG.eval()
    print(data_loader)
    for batch_id, data in enumerate(data_loader):
        print(f'data id: {batch_id}')
        ###* Prepare training data:
        text_emb, real_images = data
        text_emb = text_emb.to(args.device)
        real_images = real_images.to(args.device)

        ###* Generate fake images:
        noise.data.normal_(0, 1)
        _, fake_images, mu, logvar = netG(text_emb, noise)
    

        if batch_id % 10 == 0:
            ###* save the image result for each epoch:
            lr_fake, fake, _, _ = netG(text_emb, fixed_noise)
            util.save_img_results(real_images, fake, epoch, args)
            if lr_fake is not None:
                util.save_img_results(None, lr_fake, epoch, args)

if __name__ == "__main__":
    args_ = args.get_all_args()
    run(args_)
    
    #change this path 
    embeddings_path = '/Users/anshikaraman/Downloads/submission/STACK_GAN_Project/input/data/birds/train/char-CNN-RNN-embeddings.pickle'
    device = torch.device(args_.device)
    nz = args_.n_z
    batch_size = args_.train_bs
    noise = Variable(torch.FloatTensor(batch_size, nz)).to(device)
    with torch.no_grad():
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1)).to(device)
    training_set = dataset.CUBDataset(pickl_file=args_.train_filenames, img_dir=args_.images_dir, cnn_emb=args_.cnn_annotations_emb_train, stage=args_.STAGE)
    train_data_loader = torch.utils.data.DataLoader(training_set, batch_size=args_.train_bs, num_workers=args_.train_workers)
    sample(args_, train_data_loader, noise, fixed_noise, args_.TRAIN_MAX_EPOCH)
