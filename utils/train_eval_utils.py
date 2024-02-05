import torch
from utils.tools import tocuda
from utils.test_utils import estimate_pose_norm_kpts, estimate_pose_from_E, compute_pose_error, pose_auc
import numpy as np
from tqdm import tqdm
from models.loss import MatchLoss
from utils.distributed_utils import is_main_process, reduce_value
import sys
from utils.test_utils import outlier_metric, pose_err, pose_metric, test_sample, get_pool_result


def train_one_epoch(model, optimizer, data_loader, conf, device, epoch, cur_global_step, **kwargs):
    model.train()

    loss_function = MatchLoss(conf)
    mean_loss = torch.zeros(1).to(device)
    mean_eloss = torch.zeros(1).to(device)
    mean_closs = torch.zeros(1).to(device)
    mean_inlier = torch.zeros(1).to(device)
    optimizer.zero_grad()

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        if step == 1:
            break
        cur_global_step += step * torch.cuda.device_count()

        train_data = tocuda(data)

        cur_lr = optimizer.param_groups[0]['lr']

        xs = train_data['xs']
        ys = train_data['ys']
        img1 = train_data['img1']
        img2 = train_data['img2']

        res_logits, ys_ds, res_e_hat, _, xs_ds = model(xs, ys, img1, img2)

        loss, ess_loss, classif_loss = loss_function.run(cur_global_step, data, res_logits, ys_ds, [res_e_hat[-1]])
        loss.backward()

        with torch.no_grad():
            is_pos = (ys_ds[-1] < conf.obj_geod_th).type(ys_ds[-1].type())
            is_neg = (ys_ds[-1] >= conf.obj_geod_th).type(ys_ds[-1].type())
            inlier_ratio = torch.sum(is_pos, dim=-1) / (torch.sum(is_pos, dim=-1) + torch.sum(is_neg, dim=-1))
            inlier_ratio = inlier_ratio.mean().item()

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        ess_loss = reduce_value(torch.tensor(ess_loss).to(device), average=True)
        mean_eloss = (mean_eloss * step + ess_loss.detach()) / (step + 1)
        classif_loss = reduce_value(torch.tensor(classif_loss).to(device), average=True)
        mean_closs = (mean_closs * step + classif_loss.detach()) / (step + 1)
        inlier_ratio = reduce_value(torch.tensor(inlier_ratio).to(device), average=True)
        mean_inlier = (mean_inlier * step + inlier_ratio.detach()) / (step + 1)

        if is_main_process():
            data_loader.desc = "[Training epoch {}] LR: {}, loss: {}, ess_loss: {}, classif_loss: {}, inlier_ratio: {}, ".\
                format(epoch, cur_lr, round(mean_loss.item(), 3), round(mean_eloss.item(), 3), round(mean_closs.item(), 3), round(mean_inlier.item(), 3))
            kwargs['tb_writer'].add_scalar("train_loss", mean_loss.item(), cur_global_step + step)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, config, epoch):
    model.eval()

    torch.multiprocessing.set_sharing_strategy('file_system')

    err_ts, err_Rs = [], []
    precision, recall, f_scores = [], [], []
    results, pool_arg = [], []
    eval_step, eval_step_i, num_processor = 100, 0, 1

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, test_data in enumerate(data_loader):
        xs = test_data['xs'].cuda()
        ys = test_data['ys'].cuda()
        img1 = test_data['img1'].cuda()
        img2 = test_data['img2'].cuda()

        res_logits, ys_ds, res_e_hat, y_hat, xs_ds = model(xs, ys, img1, img2)

        if config.tag == "logit":
            logit = res_logits[-1]
            y_in = ys
            x_in = xs
            e_hat = res_e_hat[-1]
        elif config.tag == "epi":
            logit = y_hat
            y_in = ys
            x_in = xs
            e_hat = res_e_hat[-1]

        # outlier rejection metric
        pre, rec, f_s = outlier_metric(logit, config, y_in, config.tag)
        precision.append(pre)
        recall.append(rec)
        f_scores.append(f_s)

        # pose error
        pool_arg, results, eval_step_i = pose_err(e_hat, pool_arg, test_data, eval_step_i, eval_step, logit, config, results, num_processor, y_in, x_in)

        mkpts0 = xs.squeeze()[:, :2].cpu().detach().numpy()
        mkpts1 = xs.squeeze()[:, 2:].cpu().detach().numpy()

        mask = y_hat.squeeze().cpu().detach().numpy() < config.thr
        mask_kp0 = mkpts0[mask]
        mask_kp1 = mkpts1[mask]

        if config.use_ransac_auc == True:
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1)
        else:
            e_hat = e_hat[-1].view(3, 3).cpu().detach().numpy()
            ret = estimate_pose_from_E(mkpts0, mkpts1, mask, e_hat)

        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            R_gt, t_gt = test_data['Rs'], test_data['ts']
            T_0to1 = torch.cat([R_gt.squeeze(), t_gt.squeeze().unsqueeze(-1)], dim=-1).numpy()
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        err_ts.append(err_t)
        err_Rs.append(err_R)
        data_loader.desc = "[Testing epoch {}] err_t: {}, err_R: {}".format(epoch, round(err_t, 3), round(err_R, 3))

    if len(pool_arg) > 0:
        results += get_pool_result(num_processor, test_sample, pool_arg)

    measure_list = ["err_q", "err_t", "num", 'R_hat', 't_hat']
    eval_res = {}
    for measure_idx, measure in enumerate(measure_list):
        eval_res[measure] = np.asarray([result[measure_idx] for result in results])

    ret_val = pose_metric(eval_res)

    out_eval = {'error_t': err_ts, 'error_R': err_Rs}

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100. * yy for yy in aucs]

    return aucs[0], aucs[1], aucs[2], ret_val, np.mean(np.asarray(precision)), np.mean(np.asarray(recall)), np.mean(np.asarray(f_scores))