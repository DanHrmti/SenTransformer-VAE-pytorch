"""
Main function for arguments parsing
Authors: Dan Haramati, Nofit Segal
"""
# imports
import torch
import argparse
from TransformerVAE import train_TransformerVAE

if __name__ == "__main__":
    """
        Recommended hyper-parameters:
        - PTB:       e_dim: 512, z_dim: 32, batch_size: 32, optimizer: 'SGDwM', lr: 0.1, beta_min: 0.005, beta_max: 0.04
        - WikiText2: e_dim: 512, z_dim: 32, batch_size: 32, optimizer: 'SGDwM', lr: 0.1, beta_min: 0.005, beta_max: 0.04
    """
    parser = argparse.ArgumentParser(description="train TransformerVAE")

    parser.add_argument("-d", "--dataset", type=str, help="dataset to train on: ['PTB', 'WikiText2']")
    parser.add_argument("-e", "--e_dim", type=int, help="word embedding dimension", default=512)
    parser.add_argument("-a", "--nheads", type=int, help="number of attention heads in transformer encoder/decoder blocks", default=4)
    parser.add_argument("-q", "--nTElayers", type=int, help="number of transformer encoder layers", default=4)
    parser.add_argument("-p", "--nTDlayers", type=int, help="number of transformer decoder layers", default=4)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=32)
    parser.add_argument("-i", "--num_epochs", type=int, help="total number of epochs to run", default=20)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("-o", "--optim", type=str, help="optimizer for training: ['SGDwM', 'Adam', 'SGD']", default='SGDwM')
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=0.1)
    parser.add_argument("-t", "--beta_sch", type=str, help="beta scheduler: ['cyclic', 'anneal', 'constant']", default='anneal')
    parser.add_argument("-n", "--beta_min", type=float, help="minimum value of beta in scheduler", default=0.005)
    parser.add_argument("-x", "--beta_max", type=float, help="maximum value of beta in scheduler", default=0.04)
    parser.add_argument("-w", "--beta_warmup", type=int, help="number of warmup epochs beta will receive beta_min value", default=4)
    parser.add_argument("-r", "--beta_period", type=int, help="number of epochs in a period of the cyclic beta scheduler", default=8)

    parser.add_argument("-v", "--save_interval", type=int, help="epochs between checkpoint saving", default=5)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=-1)
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device", default=-1)

    args = parser.parse_args()

    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))

    train_TransformerVAE(dataset=args.dataset, e_dim=args.e_dim, nheads=args.nheads, nTElayers=args.nTElayers,
                         nTDlayers=args.nTDlayers, z_dim=args.z_dim, num_epochs=args.num_epochs,
                         batch_size=args.batch_size, optim=args.optim, lr=args.lr, beta_sch=args.beta_sch,
                         beta_min=args.beta_min, beta_max=args.beta_max, beta_warmup=args.beta_warmup,
                         beta_period=args.beta_period, save_interval=args.save_interval, device=device, seed=args.seed)
