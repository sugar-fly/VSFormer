import cv2
import numpy
import math
import numpy as np
import torch
from multiprocessing import Pool as ThreadPool


def estimate_pose_norm_kpts(kpts0, kpts1, thresh=1e-3, conf=0.99999):
    if len(kpts0) < 5:
        return None

    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=thresh, prob=conf, method=cv2.RANSAC)

    if E is None:
        return None

    assert E is not None

    best_num_inliers = 0
    new_mask = mask
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, mask_ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)

    return ret


def estimate_pose_from_E(kpts0, kpts1, mask, E):
    assert E is not None
    mask = mask.astype(np.uint8)
    E = E.astype(np.float64)
    kpts0 = kpts0.astype(np.float64)
    kpts1 = kpts1.astype(np.float64)
    I = np.eye(3).astype(np.float64)

    best_num_inliers = 0
    ret = None

    for _E in np.split(E, len(E) / 3):

        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, I, 1e9, mask=mask)

        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


def outlier_metric(logit, config, y_in, tag):
    # The groundtruth epi sqr
    gt_geod_d = y_in
    is_pos = (gt_geod_d < config.obj_geod_th).type(logit.type())
    is_neg = (gt_geod_d >= config.obj_geod_th).type(logit.type())

    if tag == "logit":
        precision = torch.mean(
            torch.sum((logit > 0).type(is_pos.type()) * is_pos, dim=1) /
            (torch.sum((logit > 0).type(is_pos.type()) * (is_pos + is_neg), dim=1) + 1e-15)
        )
        recall = torch.mean(
            torch.sum((logit > 0).type(is_pos.type()) * is_pos, dim=1) /
            (torch.sum(is_pos, dim=1) + 1e-15)
        )
    elif tag == "epi":
        precision = torch.mean(
            torch.sum((logit < 1e-4).type(is_pos.type()) * is_pos, dim=1) /
            torch.sum((logit < 1e-4).type(is_pos.type()) * (is_pos + is_neg), dim=1)
        )
        recall = torch.mean(
            torch.sum((logit < 1e-4).type(is_pos.type()) * is_pos, dim=1) /
            (torch.sum(is_pos, dim=1) + 1e-15)
        )

    f_scores = 2 * precision.item() * recall.item() / (precision.item() + recall.item() + 1e-15)

    return precision.item(), recall.item(), f_scores


def get_pool_result(num_processor, fun, args):
    pool = ThreadPool(num_processor)
    pool_res = pool.map(fun, args)
    pool.close()
    pool.join()
    return pool_res


def quaternion_from_matrix(matrix, isprecise=False):
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t


def eval_nondecompose(p1s, p2s, E_hat, dR, dt, scores, tag):
    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    if tag == "logit":
        th = np.sort(scores)[::-1][num_top]
        mask = scores < th
    elif tag == "epi":
        th = np.sort(scores)[::1][num_top]
        mask = scores >= th

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(p1s.dtype)
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E_hat, p1s_good, p2s_good)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            print("Failed in evaluation")
            print(E_hat)
            print(R)
            print(t)
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    # Change mask type
    mask = mask.flatten().astype(bool)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t


def eval_decompose(p1s, p2s, dR, dt, mask=None, method=cv2.LMEDS, probs=None):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        if probs is None and method != "MLESAC":
            # Change the threshold from 0.01 to 0.001 can largely imporve the results
            # For fundamental matrix estimation evaluation, we also transform the matrix to essential matrix.
            # This gives better results than using findFundamentalMat
            E, mask_new = cv2.findEssentialMat(p1s_good, p2s_good, method=method, threshold=0.001)
        else:
            pass
        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good, p2s_good, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t


def test_sample(args):
    _xs, _dR, _dt, _e_hat, _y_hat, _y_gt, config = args
    _xs = _xs.reshape(-1, 4).astype('float64')
    _dR, _dt = _dR.astype('float64').reshape(3,3), _dt.astype('float64')
    _y_hat_out = _y_hat.flatten().astype('float64')
    e_hat_out = _e_hat.flatten().astype('float64')

    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = _y_hat_out
    # choose top ones (get validity threshold)
    if config.tag == "logit":
        _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
        _mask_before = _valid >= max(0, _valid_th)
    elif config.tag == "epi":
        _valid_th = np.sort(_valid)[::1][config.obj_top_k]
        _mask_before = _valid < min(1e-4, _valid_th)

    if not config.use_ransac_map:
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_nondecompose(_x1, _x2, e_hat_out, _dR, _dt, _y_hat_out, config.tag)
    else:
        # actually not use prob here since probs is None
        if config.post_processing == 'RANSAC':
            method = cv2.RANSAC
        elif config.post_processing == 'PROSAC':
            method = cv2.USAC_PROSAC
        else:
            method = cv2.USAC_MAGSAC
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_decompose(_x1, _x2, _dR, _dt, mask=_mask_before, method=method, \
            probs=None, weighted=False, use_prob=True)
    if _R_hat is None:
        _R_hat = np.random.randn(3,3)
        _t_hat = np.random.randn(3,1)
    return [float(_err_q), float(_err_t), float(_num_inlier), _R_hat.reshape(1,-1), _t_hat.reshape(1,-1)]


def pose_err(e_hat, pool_arg, test_data, eval_step_i, eval_step, logit, config, results, num_processor, y_in, x_in):
    for batch_idx in range(e_hat.shape[0]):
        test_xs = x_in[batch_idx].detach().cpu().numpy()

        pool_arg += [(test_xs, test_data['Rs'][batch_idx].detach().cpu().numpy(),
                      test_data['ts'][batch_idx].detach().cpu().numpy(), e_hat[batch_idx].detach().cpu().numpy(),
                      logit[batch_idx].detach().cpu().numpy(),
                      y_in[batch_idx, :].detach().cpu().numpy(), config)]

        eval_step_i += 1
        if eval_step_i % eval_step == 0:
            results += get_pool_result(num_processor, test_sample, pool_arg)
            pool_arg = []

    return pool_arg, results, eval_step_i


def pose_metric(eval_res):

    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    cur_err_t = np.array(eval_res["err_t"]) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(cur_err_q, ths)
    t_acc_hist, _ = np.histogram(cur_err_t, ths)
    qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
    num_pair = float(len(cur_err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    t_acc_hist = t_acc_hist.astype(float) / num_pair
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    t_acc = np.cumsum(t_acc_hist)
    qt_acc = np.cumsum(qt_acc_hist)

    # Return qt_acc
    ret_val = [np.mean(qt_acc[:i]) for i in range(1, 5)]
    return ret_val