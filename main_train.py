import argparse
import torch
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import Evaluate
from codes.utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='hazelnut', type=str)
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=64, type=int)

parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

args = parser.parse_args()


def train():
    obj = args.obj
    D = args.D
    lr = args.lr
    print("STARTING")
    with task('Networks'):
        print("NETWORKS")
        enc = EncoderHier(64, D).cuda()
        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        print("DATASETS")
        # Preprocess
        ev = Evaluate()
        print("load images")
        load = LoadImages(DEFAULTOBJ)
        
        standardized_images_train = load.get_standardized_images_train()
        standardized_images_test = load.get_standardized_images_test(image)

        ev.set_x_tr(standardized_images_train)
        ev.set_x_te(standardized_images_test)

        train_x = NHWC2NCHW(x_tr)

        print("init dataset")
        rep = 100
        datasets = dict()
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)

    f = open("logs.txt", "a")
    f.write("STARTING NEW TRAINING")
    f.write("\n")
    f.close()
    print('Start training')
    for i_epoch in range(args.epochs):
        print("EPOCH: " + str(i_epoch))
        f = open("logs.txt", "a")
        f.write("EPOCH: " + str(i_epoch))
        f.write("\n")
        f.close()
        if i_epoch != 0:
            for module in modules:
                module.train()

            for d in loader:
                d = to_device(d, 'cuda', non_blocking=True)
                opt.zero_grad()

                loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
                loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

                loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

                
                f = open("logs.txt", "a")
                f.write(str(loss))
                f.write("\n")
                f.close()

                
                
                loss.backward()
                opt.step()

        ev.eval_encoder_NN_multiK(enc)
        enc.save(obj)

if __name__ == '__main__':
    train()
