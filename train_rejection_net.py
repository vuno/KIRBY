import argparse
import os
import torch
import glob
import torch.nn.functional as F
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from models.wrn import WideResNet
from models.ood_detector import OODDector
from tqdm import tqdm
from torchvision.datasets import SVHN
from utils import Accumulator, get_ood_score, get_ood_performance, set_seed
from data_loader import CIFAR10withOOD, CIFAR100withOOD


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_list = [
    trn.ToTensor(),
    trn.Normalize(mean, std),
]
transform_train = trn.Compose(train_list)
ind_test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

# augmentation
ood_transform = [
    trn.RandomHorizontalFlip(),
    trn.RandomAffine(0, (0.0, 0.0), (1.0, 1.5)),
    trn.RandomAutocontrast(0.2),
    trn.RandomInvert(0.2),
    trn.ToTensor(),
    trn.Normalize(mean, std),
]
transform_ood = trn.Compose(ood_transform)


def train_ood_detector(args, train_ood_dir=None, th=0.3, save_dir=None, ood_loader=None):
    if 'cifar10' == args.dataset:
        ind_test_data = dset.CIFAR10(f'{args.torchdata_root}/cifarpy', download=True,
                                     train=False, transform=ind_test_transform)
        num_classes = 10
    else:
        ind_test_data = dset.CIFAR100(f'{args.torchdata_root}/cifarpy', download=True,
                                      train=False, transform=ind_test_transform)
        num_classes = 100
    ind_test_loader = DataLoader(ind_test_data, batch_size=args.batch_size // 2, shuffle=False, num_workers=4)

    # load pre-trained model and ood classifier
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).eval()
    net.load_state_dict(torch.load(f"./model_ckpt/wrn/{args.dataset}_wrn_pretrained_epoch_99.pt"))

    if args.multi_class:
        multi_class = True
        criterion = torch.nn.CrossEntropyLoss()
    else:
        multi_class = False
        criterion = F.binary_cross_entropy_with_logits
    ood_detector = OODDector(num_classes=num_classes, multi_class=multi_class)

    optimizer = torch.optim.SGD(ood_detector.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if torch.cuda.is_available():
        net.cuda()
        ood_detector.cuda()

    valid_image_files = glob.glob(os.path.join(train_ood_dir, f"*_{str(th)}_train.png"))
    print(f"CAM-based OOD NUM : {len(valid_image_files)}")
    # ood train dataset
    if args.dataset == 'cifar10':
        concat_dataset = CIFAR10withOOD(f'{args.torchdata_root}/cifarpy', args.batch_size, ood_files=valid_image_files,
                                        download=True, train=True, transform=transform_train, ood_transform=transform_ood,
                                        use_patch_aug=True, multiclass=multi_class)
    else:
        concat_dataset = CIFAR100withOOD(f'{args.torchdata_root}/cifarpy', args.batch_size, ood_files=valid_image_files,
                                         download=True, train=True, transform=transform_train, ood_transform=transform_ood,
                                         use_patch_aug=True, multiclass=multi_class)
    train_loader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # train ood classifier
    for epoch in range(args.epochs):
        metrics = Accumulator()
        loader = tqdm(train_loader, disable=False, ascii=True)
        loader.set_description('[%s %03d/%03d]' % ('train', epoch + 1, args.epochs))
        cnt = 0
        for batch_idx, (x_data, y_data) in enumerate(loader):
            cnt += args.batch_size

            if torch.cuda.is_available():
                x_data = x_data.cuda()
                y_data = y_data.cuda()

            optimizer.zero_grad()
            features = net.get_all_blocks(x_data)
            pred = ood_detector(features.detach())
            if multi_class:
                loss = criterion(pred, y_data.view(-1).cuda().to(torch.long))
            else:
                y_data = torch.where(y_data == num_classes, torch.ones_like(y_data), torch.zeros_like(y_data))
                loss = criterion(pred, y_data.view(-1, 1).cuda().to(torch.float32))

            metrics.add_dict({
                'loss': loss.item() * args.batch_size,
            })
            postfix = metrics / cnt
            loader.set_postfix(postfix)
            loss.backward()
            optimizer.step()

        scheduler.step()

    ind_scores = get_ood_score(net, ood_detector, ind_test_loader, multi_class)
    svhn_scores = get_ood_score(net, ood_detector, ood_loader, multi_class)

    with open(os.path.join(save_dir, args.ood_method), 'w') as f:
        auroc, aupr, fpr = get_ood_performance(-ind_scores, -svhn_scores)
        print(f"SVHN ({args.ood_method}) : AUROC : {auroc}, AUPR : {aupr}, FPR : {fpr}")
        f.write(f"SVHN ({args.ood_method}) : AUROC : {auroc}, AUPR : {aupr}, FPR : {fpr} \n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default="0", type=str)
    parser.add_argument('--cam_method', default='layercam', type=str, help='gradcam, gradcampp, layercam')
    parser.add_argument('--ood_method', default='kirby')
    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 cifar100')
    parser.add_argument('--surrogate_ood_dir', default='./surrogate_ood_datasets')
    parser.add_argument('--torchdata_root', default='./torchdata', type=str)

    # Loading details
    parser.add_argument('--multi_class', default=1, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    # wide-resnet params
    parser.add_argument('--architecture', default='wrn')
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_seed(42)

    IMG_SIZE = 32
    CAM_TH = 0.3
    CAM_BLOCK = 2
    train_ood_dir = f"{args.surrogate_ood_dir}/{args.architecture}/{args.dataset}/{args.cam_method}/layer_{CAM_BLOCK+1}/ood_sample/*/"

    if args.multi_class:
        args.ood_method = 'kirby_m'
    else:
        args.ood_method = 'kirby_b'

    save_dir = f'./kirby_result_{args.architecture}/{args.dataset}/{args.ood_method}'
    os.makedirs(save_dir, exist_ok=True)

    # /////////////// SVHN ///////////////
    svhn_data = SVHN(root=f'{args.torchdata_root}/svhn/', split="test",
                     transform=trn.Compose([trn.Resize((IMG_SIZE, IMG_SIZE)), trn.ToTensor(), trn.Normalize(mean, std)]),
                     download=True)
    svhn_loader = DataLoader(svhn_data, batch_size=args.batch_size, pin_memory=True, drop_last=False)
    train_ood_detector(args, train_ood_dir, CAM_TH, save_dir, svhn_loader)
