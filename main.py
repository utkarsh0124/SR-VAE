""" PyTorch implimentation of VAE and Super-Resolution VAE.

    Reposetory Author:
        Ioannis Gatopoulos, 2020
"""
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
import torch
import os
import torchvision.models as models
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm

from src import *

def my_dataloader(batch_size):
    path = r'/content/drive/MyDrive/CUB200_2011/SampleImage/SampleImage'
    
    train_path = os.path.join(path,'train')
    test_path = os.path.join(path, 'test')
    valid_path = os.path.join(path, 'validation')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess data
    transform = transforms.Compose([transforms.Resize(size=(32,32)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    
    print('Train Path', train_path)
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    validate_dataset = datasets.ImageFolder(valid_path, transform=transform)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size)
    
    return train_dataloader, validate_dataloader, test_dataloader
    

def train_model(dataset, model, writer=None):
    train_loader, valid_loader, test_loader = my_dataloader(batch_size=64)
    data_shape = get_data_shape(train_loader)

    model = nn.DataParallel(globals()[model](data_shape).to(args.device))
    model.module.initialize(train_loader)  

    criterion = ELBOLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-7)
    scheduler = LowerBoundedExponentialLR(optimizer, gamma=0.999999, lower_bound=0.0001)

    n_parameters(model, writer)

    for epoch in tqdm(range(1, args.epochs)):
        # Train and Validation epoch
        train_losses = train(model, criterion, optimizer, scheduler, train_loader)
        valid_losses = evaluate(model, criterion, valid_loader)
        # Visual Evaluation
        generate(model, args.n_samples, epoch, writer)
        reconstruction(model, valid_loader, args.n_samples, epoch, writer)
        # Saving Model and Loggin
        is_saved = save_model(model, optimizer, valid_losses['nelbo'], epoch)
        logging(epoch, train_losses, valid_losses, is_saved, writer)


def load_and_evaluate(dataset, model, writer=None):
    pth = r'/content/drive/MyDrive/srVAE/src/models'

    # # configure paths
    pth = os.path.join(pth, 'pretrained', args.model, args.dataset)
    pth_inf = os.path.join(pth, 'inference', 'model.pth')
    pth_train = os.path.join(pth, 'trainable', 'model.pth')

    # get data
    train_loader, valid_loader, test_loader = my_dataloader(batch_size=64)
    data_shape = get_data_shape(train_loader)

    # deifine model
    model = globals()[model](data_shape).to(args.device)
    model.initialize(train_loader)

    # load trained weights for inference
    if os.path.exists(pth_train):
        checkpoint = torch.load(pth_train)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Model successfully loaded!')
        except RuntimeError:
            print('* Failed to load the model. Parameter mismatch.')
            quit()
    model = nn.DataParallel(model).to(args.device)
    model.eval()
    criterion = ELBOLoss()
    
    # Evaluation of the model
    # --- calculate elbo ---
    test_losses = evaluate(model, criterion, test_loader)
    print('ELBO: {} bpd'.format(test_losses['bpd']))

    # --- image generation ---
    generate(model, n_samples=15*15)

    # --- image reconstruction ---
    reconstruction(model, test_loader, n_samples=15)

    # --- image interpolation ---
    interpolation(model, test_loader, n_samples=15)

    # --- calculate nll ---
    bpd = calculate_nll(model, test_loader, criterion, args, iw_samples=args.iw_test)
    print('NLL with {} weighted samples: {:4.2f}'.format(args.iw_test, bpd))


# ----- main -----

def main():
    # Print configs
    print_args(args)

    # Control random seeds
    fix_random_seed(seed=args.seed)

    # Initialize TensorBoad writer (if enabled)
    writer = None
    if args.use_tb:
        writer = SummaryWriter(log_dir='./logs/'+args.dataset+'_'+args.model+'_'+args.tags +
                               datetime.now().strftime("/%d-%m-%Y/%H-%M-%S"))
        writer.add_text('args', namespace2markdown(args))

    # print(args)

    # Train model
    train_model(args.dataset, args.model, writer)

    # Evaluate best (latest saved) model
    load_and_evaluate(args.dataset, args.model, writer)

    # End Experiment
    writer.close()
    print('\n'+24*'='+' Experiment Ended '+24*'=')


# ----- python main.py -----

if __name__ == "__main__":
    main()
