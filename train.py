import os
import argparse
import datetime
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import transforms as T

from my_dataset import PotsdamSegmentation

from model.CTFuse import CTFuse

# from config import get_config
from utils import train_one_epoch_seg, evaluate_seg,create_lr_scheduler
from torch.backends import cudnn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.cuda.manual_seed(0)
cudnn.enabled = True
cudnn.benchmark = True




def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    if os.path.exists("./seg_weights/{}".format(args.save_path)) is False:
        os.makedirs("./seg_weights/{}".format(args.save_path))

    tb_writer = SummaryWriter()

    # 用来保存训练以及验证过程中信息
    results_file = "./seg_weights/{}/results{}.txt".format(args.save_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": T.Compose([T.RandomCrop(256),
                            #T.RandomHorizontalFlip(0.5),
                            #T.RandomVerticalFlip(0.5),
                            T.ToTensor(),
                            T.Normalize([0.3412, 0.3637, 0.3378], [0.1402, 0.1384, 0.1439])]),
        # "val": T.Compose([T.Resize(256),
        #                            T.CenterCrop(224),
        #                            T.ToTensor(),
        #                            T.Normalize([0.3412, 0.3637, 0.3378], [0.1402, 0.1384, 0.1439])])}
        "val": T.Compose([T.Resize(256),
                          T.CenterCrop(256),
                          T.ToTensor(),
                          T.Normalize([0.3412, 0.3637, 0.3378], [0.1402, 0.1384, 0.1439])])}


    # 实例化Potsdam训练数据
    train_dataset = PotsdamSegmentation(args.data_path,
                                    transforms=data_transform['train'],
                                    txt_name="train")

    val_dataset = PotsdamSegmentation(args.data_path,
                                  transforms=data_transform['val'],
                                  txt_name="test")
    print("Data initialization is finish!")


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model=CTFuse(n_classes=6).to(device)
    # 输出模型所有key值
    # for key, value in model.named_parameters():
    #     print(key)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    #optimizer = optim.Adam(pg, lr=args.lr, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    max_acc = 0
    fig_train_acc = []
    fig_val_acc = []
    start_time = time.time()


    for epoch in range(args.epochs):
        #train
        mean_loss, lr = train_one_epoch_seg(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=epoch,
                                            lr_scheduler=lr_scheduler,
                                            print_freq=1000,
                                            scaler=scaler)

        # validate
        # 在训练集上评估
        if epoch%5==0:
            confmat1 = evaluate_seg(model, train_loader, device=device, num_classes=args.num_classes)
            val_info1 = str(confmat1)
            print(val_info1)
            fig_train_acc.append(float(val_info1[-5:]))
        else:
            fig_train_acc.append(float(0))

        # 在验证集上评估
        confmat = evaluate_seg(model=model, data_loader=val_loader, device=device, num_classes=args.num_classes)
        val_info = str(confmat)
        print(val_info)
        fig_val_acc.append(float(val_info[-5:]))

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")
        if float(val_info[-5:]) > max_acc:
            max_acc=float(val_info[-5:])
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "seg_weights/{}/model_{}_{}.pth".format(args.save_path,epoch,float(val_info[-5:])))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    x = range(len(fig_train_acc))
    plt.plot(x, fig_train_acc, label='train')
    plt.plot(x, fig_val_acc, label='val')
    plt.legend()
    plt.savefig('./seg_weights/{}/accuracy.png'.format(args.save_path))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="./dataset_vh_256")
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./pretrain/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--save-path', default="MAnet_pretrain", help='The path to save the loss graph')
    # Mixed precision training parameters
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    opt = parser.parse_args()

    main(opt)

