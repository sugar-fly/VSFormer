from config import Config
import os
import torch
from datasets.CorresDataset import CorrespondencesDataset, collate_fn
from torch.utils.data import DataLoader
from models.vsformer import VSFormer
import torch.optim as optim
from utils.train_eval_utils import train_one_epoch, evaluate
from utils.distributed_utils import init_distributed_mode, dist, cleanup
import tempfile
import random
import time
from torch.utils.tensorboard import SummaryWriter
from utils.tools import safe_load_weights


if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()
    init_distributed_mode(args=conf)

    rank = conf.rank
    device = torch.device(conf.device)
    world_size = 2
    batch_size = 20
    true_batch_size = world_size * batch_size
    _scaling = true_batch_size / conf.canonical_bs
    true_lr = conf.canonical_lr * _scaling
    conf.loss_essential_init_iter = int(conf.canonical_bs * 20000 // true_batch_size)
    checkpoint_path = ""

    if rank == 0:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir=conf.writer_dir)

    sleeptime = random.randint(0, 5)
    time.sleep(sleeptime)
    if not os.path.isdir(conf.checkpoint_path):
        os.makedirs(conf.checkpoint_path)
    if not os.path.isdir(conf.best_model_path):
        os.makedirs(conf.best_model_path)

    train_dataset = CorrespondencesDataset(conf.data_tr, conf)
    valid_dataset = CorrespondencesDataset(conf.data_va, conf)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    if rank == 0:
        print('Using {} dataloader workers every process'.format(conf.num_workers))

    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_batch_sampler,
                              pin_memory=True,
                              num_workers=conf.num_workers,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=8,
                              collate_fn=collate_fn)

    model = VSFormer(conf).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=true_lr, weight_decay=conf.weight_decay)

    best_auc = -1
    start_epoch = -1

    if os.path.exists(conf.resume):
        weights_dict = torch.load(conf.resume, map_location=device)
        best_auc = weights_dict['best_auc']
        start_epoch = weights_dict['epoch']
        safe_load_weights(model, weights_dict['state_dict'])
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[conf.gpu], find_unused_parameters=True)

    for epoch in range(start_epoch + 1, conf.epochs):
        train_sampler.set_epoch(epoch)
        cur_global_step = (epoch - 1) * train_dataset.__len__()

        if rank == 0:
            mean_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device="cuda",
                                        epoch=epoch,
                                        conf=conf,
                                        cur_global_step=cur_global_step,
                                        tb_writer=tb_writer)
        else:
            mean_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device="cuda",
                                        epoch=epoch,
                                        conf=conf,
                                        cur_global_step=cur_global_step)

        if rank == 0:
            aucs5, aucs10, aucs20, va_res, precisions, recalls, f_scores = evaluate(model, valid_loader, conf, epoch=epoch)

            print("[AUC result epoch {}]  AUC@5: {}  AUC@10: {}  AUC@20: {}".format(epoch, round(aucs5, 3), round(aucs10, 3), round(aucs20, 3)))
            print("[Pose metric epoch {}]  mAP5: {}  mAP10: {}  mAP20: {}".format(epoch, round(va_res[0] * 100, 3), round(va_res[1] * 100, 3), round(va_res[3] * 100, 3)))
            print("[Outlier metric epoch {}]  Precisions: {}  Recalls: {}  F_scores: {}\n".format(epoch, round(precisions * 100, 3), round(recalls * 100, 3), round(f_scores * 100, 3)))

            tags = ["train_loss", "AUC@5", "AUC@10", "AUC@20", "mAP5", "mAP10", "mAP20", "Precisions", "Recalls", "F_scores", "learning_rate"]
            tb_writer.add_scalar(tags[1], aucs5, epoch)
            tb_writer.add_scalar(tags[2], aucs10, epoch)
            tb_writer.add_scalar(tags[3], aucs20, epoch)
            tb_writer.add_scalar(tags[4], va_res[0] * 100, epoch)
            tb_writer.add_scalar(tags[5], va_res[1] * 100, epoch)
            tb_writer.add_scalar(tags[6], va_res[3] * 100, epoch)
            tb_writer.add_scalar(tags[7], precisions * 100, epoch)
            tb_writer.add_scalar(tags[8], recalls * 100, epoch)
            tb_writer.add_scalar(tags[9], f_scores * 100, epoch)
            tb_writer.add_scalar(tags[10], optimizer.param_groups[0]["lr"], epoch)

            if aucs5 > best_auc:
                print("Saving best model with auc5 = {}\n".format(aucs5))
                best_auc = aucs5
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'best_auc': best_auc,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(conf.best_model_path, 'model_best.pth'))

            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(conf.checkpoint_path, 'checkpoint{}.pth'.format(epoch)))

    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()