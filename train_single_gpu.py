import os
import torch
from config import Config
from torch.utils.tensorboard import SummaryWriter
from datasets.CorresDataset import CorrespondencesDataset, collate_fn
from torch.utils.data import DataLoader
from models.vsformer import VSFormer
import torch.optim as optim
from utils.train_eval_utils import train_one_epoch, evaluate
from utils.tools import safe_load_weights


if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()

    batch_size = 20
    true_batch_size = 1 * batch_size
    _scaling = true_batch_size / conf.canonical_bs
    true_lr = conf.canonical_lr * _scaling

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir=conf.writer_dir)

    if not os.path.isdir(conf.checkpoint_path):
        os.makedirs(conf.checkpoint_path)
    if not os.path.isdir(conf.best_model_path):
        os.makedirs(conf.best_model_path)

    train_dataset = CorrespondencesDataset(conf.data_tr, conf)
    valid_dataset = CorrespondencesDataset(conf.data_va, conf)

    print('Using {} dataloader workers every process'.format(conf.num_workers))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=conf.num_workers,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=conf.num_workers,
                              collate_fn=collate_fn)

    model = VSFormer(conf).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=true_lr, weight_decay=conf.weight_decay)

    best_auc = -1
    start_epoch = -1

    if os.path.exists(conf.resume):
        weights_dict = torch.load(conf.resume, map_location="cuda")
        best_auc = weights_dict['best_auc']
        start_epoch = weights_dict['epoch']
        safe_load_weights(model, weights_dict['state_dict'])

    for epoch in range(start_epoch + 1, conf.epochs):
        cur_global_step = (epoch - 1) * train_dataset.__len__()

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device="cuda",
                                    epoch=epoch,
                                    conf=conf,
                                    cur_global_step=cur_global_step,
                                    tb_writer=tb_writer)

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
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(conf.best_model_path, 'model_best.pth'))

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_auc': best_auc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.checkpoint_path, 'checkpoint{}.pth'.format(epoch)))