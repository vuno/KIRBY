import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from torchcam.methods import GradCAM, GradCAMpp, LayerCAM
from models.wrn import WideResNet
from tqdm import tqdm


def generate_cifar_ood_dataset(args):
    # mean and standard deviation of channels 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    img_size = 32
    input_shape = (3, img_size, img_size)
    if 'cifar10' == args.dataset:
        train_data = dset.CIFAR10(f'{args.torchdata_root}/cifarpy', download=True, train=True, transform=test_transform)
        num_classes = 10
    else:
        train_data = dset.CIFAR100(f'{args.torchdata_root}/cifarpy', download=True, train=True, transform=test_transform)
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)

    # Load pretrained model
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate, dataset=args.dataset).eval()
    net.load_state_dict(torch.load(f"./model_ckpt/wrn/{args.dataset}_wrn_pretrained_epoch_99.pt"))
    layers = [net.block1, net.block2, net.block3]
    if torch.cuda.is_available():
        net.cuda()

    # Cam-based module
    if args.method == 'gradcam':
        localize_net = GradCAM(net, target_layer=layers, input_shape=input_shape)
    elif args.method == 'gradcampp':
        localize_net = GradCAMpp(net, target_layer=layers, input_shape=input_shape)
    elif args.method == 'layercam':
        localize_net = LayerCAM(net, target_layer=layers, input_shape=input_shape)
    else:
        raise NotImplementedError

    save_dir = os.path.join(f'./{args.surrogate_ood_dir}/{args.model}/{args.dataset}/{args.method}/')
    for layer_idx in [0, 1, 2]:
        layer_save_dir = os.path.join(save_dir, f"layer_{layer_idx + 1}", "ood_sample")

        for class_idx in range(num_classes):
            os.makedirs(os.path.join(layer_save_dir, str(class_idx)), exist_ok=True)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        # Preprocess your data and feed it to the model
        out = net(data)
        # Retrieve the CAM by passing the class index and the model output
        _activation_map = localize_net(out.squeeze(0).argmax().item(), out)

        _activation_map = [_activation_map[i].squeeze().detach().cpu().numpy() for i in range(3)]
        activation_map = []
        for i in range(3):
            if _activation_map[i].shape[0] != img_size:
                x = cv2.resize(_activation_map[i], (img_size, img_size))
            else:
                x = _activation_map[i]
            activation_map.append(x)

        class_idx = target.detach().cpu().numpy().flatten()[0]

        for layer_idx in [0, 1, 2]:
            path = os.path.join(save_dir, f"layer_{layer_idx + 1}")

            x_data_array = np.transpose(data.detach().cpu().numpy(), [0, 2, 3, 1])
            origin_x_data = (x_data_array * np.array(std).reshape([1, 1, 1, 3])) + np.array(mean).reshape([1, 1, 1, 3])
            origin_x_data = np.uint8(origin_x_data * 255)[0]

            # fast marching inpaint
            background_mask = np.uint8(activation_map[layer_idx] < args.cam_lambda)
            remove_image = np.copy(origin_x_data) * np.expand_dims(background_mask, axis=-1)
            target_mask = -1 * (background_mask.astype(np.float32) - 1.)

            inpaint = cv2.inpaint(remove_image, target_mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
            save_ood_train_path = os.path.join(path, "ood_sample", str(class_idx), f"{batch_idx}_{class_idx}_{str(args.cam_lambda)}_train.png")
            save_ood_mask_path = os.path.join(path, "ood_sample", str(class_idx), f"{batch_idx}_{class_idx}_{str(args.cam_lambda)}_mask.png")

            cv2.imwrite(save_ood_mask_path, (target_mask * 255).astype(np.uint8))
            cv2.imwrite(save_ood_train_path, inpaint[..., ::-1].astype(np.uint8))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default="0", type=str)
    parser.add_argument('--model', default='wrn', type=str)
    parser.add_argument('--method', default='layercam', type=str, help='gradcam, gramcampp, layercam')
    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 cifar100')
    parser.add_argument('--cam_lambda', default=0.3, type=float)
    parser.add_argument('--surrogate_ood_dir', default='./surrogate_ood_datasets')
    parser.add_argument('--torchdata_root', default='./torchdata')

    # Loading details
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    generate_cifar_ood_dataset(args)
