import torch
import numpy as np
import hydra
hydra.output_subdir = None

from omegaconf import DictConfig

from OCTraN.model.OCTraN import OCTraN
from OCTraN.utils.colored_logging import log as logging

from OCTraN.data.semantic_kitti.kitti_dm import KittiDataModule
from OCTraN.data.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names,
)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info(f'Using device {device}')
logging.info(f'torch.__version__ {torch.__version__}') # 1.13.0
logging.info(f'torch.version.cuda {torch.version.cuda}') # => 11.7
logging.info(f'torch.backends.cudnn.version() {torch.backends.cudnn.version()}') # => 8500
logging.info(f'torch.cuda.device_count() {torch.cuda.device_count()}')


dataset="kitti"

n_relations=4

enable_log=False
kitti_root='/data/dataset/KITTI_Odometry_Semantic'
kitti_preprocess_root='/data/dataset/kitti_semantic_preprocess'
kitti_logdir='/OCTraN/logs/'

fp_loss=True 
frustum_size=8 
batch_size=1
n_gpus=1
num_workers_per_gpu=0
exp_prefix="exp"
run=1
lr=0.01
weight_decay=0.001

context_prior=True

relation_loss=True 
CE_ssc_loss=True
sem_scal_loss=True
geo_scal_loss=True

project_1_2=True
project_1_4=True
project_1_8=True

# @hydra.main(config_name="../config/config.yaml")
def main():
    # print(config)
    exp_name = exp_prefix
    exp_name += "_{}_{}".format(dataset, run)
    exp_name += "_FrusSize_{}".format(frustum_size)
    exp_name += "_nRelations{}".format(n_relations)
    exp_name += "_WD{}_lr{}".format(weight_decay, lr)
    print(exp_name)

    class_names = kitti_class_names
    max_epochs = 30
    logdir = kitti_logdir
    full_scene_size = (256, 256, 32)
    project_scale = 2
    feature = 64
    n_classes = 20

    project_res = ["1_8"]

    class_names = kitti_class_names
    class_weights = torch.from_numpy(
        1 / np.log(semantic_kitti_class_frequencies + 0.001)
    )
    semantic_kitti_class_frequencies_occ = np.array(
        [
            semantic_kitti_class_frequencies[0],
            semantic_kitti_class_frequencies[1:].sum(),
        ]
    )
    class_weights_occ = torch.from_numpy(
        1 / np.log(semantic_kitti_class_frequencies_occ + 0.001)
    )

    # Model
    # Initialize OCTraN model
    model = OCTraN(
        frustum_size=frustum_size,
        project_scale=project_scale,
        n_relations=n_relations,
        fp_loss=fp_loss,
        embeding_dim=32,
        full_scene_size=full_scene_size,
        project_res=project_res,
        n_classes=n_classes,
        class_names=class_names,
        context_prior=context_prior,
        relation_loss=relation_loss,
        CE_ssc_loss=CE_ssc_loss,
        sem_scal_loss=sem_scal_loss,
        geo_scal_loss=geo_scal_loss,
        lr=lr,
        weight_decay=weight_decay,
        class_weights=class_weights,
    )
    model.to(device)

    # Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Load Data
    data_module = KittiDataModule(
        root='/OCTraN/dataset/semantic_kitti/data_odometry_voxels',
        preprocess_root='/OCTraN/dataset/semantic_kitti/preprocess',
        frustum_size=8,
        project_scale=2,
        batch_size=batch_size,
        num_workers=num_workers_per_gpu
    )

    logging.info(data_module)

    logger = TensorBoardLogger(save_dir='/OCTraN/logs', name='test', version="")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callbacks = [
        ModelCheckpoint(
            save_last=True,
            monitor="val/mIoU",
            save_top_k=1,
            mode="max",
            filename="{epoch:03d}-{val/mIoU:.5f}",
        ),
        lr_monitor,
    ]

    # Train from scratch
    trainer = Trainer(
        callbacks=checkpoint_callbacks,
        sync_batchnorm=True,
        deterministic=False,
        max_epochs=10,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        accelerator="cuda",
    )

    trainer.fit(model, data_module)

    # example_img = torch.rand(3,3,512,128,device=device)
    # example_vox = torch.randint(0,259,(3,32,32,4),dtype=torch.float32,device=device)

    # out = model(example_img,example_vox)
    # logging.info(out.mean())

if __name__ == "__main__":
    main()