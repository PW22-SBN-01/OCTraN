import torch
import numpy as np

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

# Model
model = OCTraN(0.001)
model.to(device)

# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load Data
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

data_module = KittiDataModule(
    root='/OCTraN/dataset/semantic_kitti/data_odometry_voxels',
    preprocess_root='/OCTraN/dataset/semantic_kitti/preprocess',
    frustum_size=8,
    project_scale=2,
    batch_size=int(1),
    num_workers=int(0)
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
    log_every_n_steps=10,
    accelerator="cuda",
)

trainer.fit(model, data_module)

example_img = torch.rand(3,3,512,128,device=device)
example_vox = torch.randint(0,259,(3,32,32,4),dtype=torch.float32,device=device)

out = model(example_img,example_vox)
