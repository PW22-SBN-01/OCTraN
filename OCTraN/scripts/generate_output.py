from OCTraN.model.OCTraN import OCTraN
from OCTraN.data.semantic_kitti.kitti_dm import KittiDataModule

import torch
import numpy as np
import os
from tqdm import tqdm
import pickle



# @hydra.main(config_name="../config/monoscene.yaml")
# def main(config: DictConfig):
def main(args):
    torch.set_grad_enabled(False)

    dataset = "kitti"
    output_path = "outputs/OCTraN_Semantic"
    batch_size = 1

    # Load pretrained models
    model_path = os.path.expanduser(args.model_path)

    # Setup dataloader
    if dataset == "kitti":
        feature = 64
        project_scale = 2
        full_scene_size = (256, 256, 32)

        data_module = KittiDataModule(
            root='/OCTraN/dataset/semantic_kitti/data_odometry_voxels',
            preprocess_root='/OCTraN/dataset/semantic_kitti/preprocess',
            frustum_size=8,
            project_scale=2,
            batch_size=batch_size,
            num_workers=0
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
        # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
    
    else:
        print("dataset not support")

    model = OCTraN.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        full_scene_size=full_scene_size,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data 
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(output_path, dataset)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch["img"] = batch["img"].cuda()
            batch["vox"] = torch.tensor(batch["CP_mega_matrices"][0]).to(dtype=torch.float32).cuda()
            if len(batch["vox"].shape) == 3:
                batch["vox"] = batch["vox"].unsqueeze(0)

            img = batch["img"]
            vox = batch["vox"]

            pred = model(img, vox)
            y_pred = torch.softmax(pred, dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            for i in range(batch_size):
                out_dict = {"y_pred": y_pred[i].astype(np.uint16)}
                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )

            
                write_path = os.path.join(output_path, batch["sequence"][i])
                filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                out_dict["fov_mask_1"] = (
                    batch["fov_mask_1"][i].detach().cpu().numpy()
                )
                out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                out_dict["T_velo_2_cam"] = (
                    batch["T_velo_2_cam"][i].detach().cpu().numpy()
                )

                os.makedirs(write_path, exist_ok=True)
                with open(filepath, "wb") as handle:
                    pickle.dump(out_dict, handle)
                    print("wrote to", filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # model_path
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to the trained model",
    )

    args = parser.parse_args()

    main(args)