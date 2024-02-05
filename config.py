import os
from datetime import datetime


class Config(object):

    def __init__(self):
        self.model = "vsformer"

        # pre_processing
        self.pre_processing = 'sift-2000'

        # data related
        self.dataset = 'yfcc100m'
        self.data_te = './data_dump/yfcc-sift-2000-test.hdf5'
        self.data_tr = './data_dump/yfcc-sift-2000-train.hdf5'
        self.data_va = './data_dump/yfcc-sift-2000-val.hdf5'
        self.image_H = 120
        self.image_W = 160

        # model related
        self.use_fundamental = False  # train fundamental matrix estimation
        self.use_ratio = 0  # use ratio test. 0: not use, 1: use before network, 2: use as side information
        self.use_mutual = 0  # use mutual nearest neighbor check. 0: not use, 1: use before network, 2: use as side information
        self.ratio_test_th = 0.8  # 0.8, ratio test threshold
        self.sr = 0.5
        self.net_channels = 128
        self.space_dim = (self.image_H // 4) * (self.image_W // 4)

        # loss related
        self.geo_loss_margin = 0.1  # clamping margin in geometry loss
        self.ess_loss_margin = 0.1  # clamping margin in contrastive loss
        self.loss_classif = 1.0  # weight of the classification loss
        self.loss_essential = 0.5  # weight of the essential loss
        self.weight_decay = 0  # l2 decay
        self.momentum = 0.9

        self.obj_geod_th = 1e-4  # threshold for the good geodesic distance
        self.thr = 3e-5
        self.obj_top_k = -1

        # training related
        self.num_workers = 8
        self.canonical_bs = 32
        self.canonical_lr = 1e-3
        self.writer_dir = os.path.join('runs', datetime.now().strftime("[" + self.model + "]-" + "[%Y_%m_%d]-[%H_%M_%S]-[debugging]"))
        self.epochs = 29  # yfcc: 29 epochs is approximately equal to 500k iterations; sun3d: 16 epochs is approximately equal to 500k iterations
        self.checkpoint_path = './checkpoint/' + self.model + '/'
        self.resume = './checkpoint/' + self.model + '/checkpoint999.pth'
        if self.use_fundamental:
            self.best_model_path = './best_model/' + self.model + '/' + self.dataset + '/fundamental/' + self.pre_processing + '/'
        else:
            self.best_model_path = './best_model/' + self.model + '/' + self.dataset + '/essential/' + self.pre_processing + '/'

        # testing related
        self.use_ransac_auc = False
        self.use_ransac_map = False
        self.tag = 'epi'  # logit or epi
        self.post_processing = 'RANSAC'

        # loss related
        self.loss_essential_init_iter = int(self.canonical_bs * 20000 // self.canonical_bs)

        # multi gpu info
        self.device = 'cuda'
        self.rank = 0
        self.world_size = 1
        self.gpu = 0
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'