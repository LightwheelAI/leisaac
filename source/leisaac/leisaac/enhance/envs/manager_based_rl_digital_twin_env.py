import cv2
import torch
import torch.nn.functional as F
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

from .manager_based_rl_digital_twin_env_cfg import ManagerBasedRLDigitalTwinEnvCfg
from .mdp import overlay_image


class ManagerBasedRLDigitalTwinEnv(ManagerBasedRLEnv):
    rgb_overlay_images: dict[str, torch.Tensor] = {}

    foreground_semantic_id_mapping: dict[str, int] = {}

    def __init__(self, cfg: ManagerBasedRLDigitalTwinEnvCfg, **kwargs):
        """
        Initialize the ManagerBasedRLDigitalTwinEnv.

        Args:
            cfg: The configuration for the ManagerBasedRLDigitalTwinEnv.
        """

        cfg = self.__setup_camera_and_foreground(cfg)

        super().__init__(cfg, **kwargs)

        self.__record_semantic_id_mapping(cfg)

    def __read_overlay_image(self, path: str, target_size: tuple[int, int]) -> torch.Tensor:
        """
        Read the overlay image and resize it to the target size.
        Args:
            path: the path to the overlay image.
            target_size: the target size of the overlay image.(width, height)
        Returns:
            the resized overlay image.(C, H, W)
        """
        image = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

        if image.dim() == 3 and image.shape[2] in [3, 4]:  # [H, W, C]
            image = image.permute(2, 0, 1)  # [C, H, W]

        resize_image = F.interpolate(
            image.unsqueeze(0), size=(target_size[1], target_size[0]), mode="bilinear"
        ).squeeze(0)
        resize_image = resize_image.squeeze(0)
        # reorder the image to [C, H, W]
        if resize_image.shape[0] in [3, 4]:  # [C, H, W]
            resize_image = resize_image.permute(1, 2, 0)  # [H, W, C]

        return resize_image

    def __setup_camera_and_foreground(self, cfg: ManagerBasedRLDigitalTwinEnvCfg) -> ManagerBasedRLDigitalTwinEnvCfg:
        """Setup the camera for the ManagerBasedRLDigitalTwinEnv.
        1. add semantic tags to the render objects
        2. add semantic segmentation to the camera data types.
        3. modify the observation cfg to add overlay_image.
        """
        for obj in cfg.render_objects:
            obj_cfg = getattr(cfg.scene, obj.name)
            obj_cfg.spawn.semantic_tags = [("class", "foreground")]

        if cfg.rgb_overlay_paths is not None:
            for camera_name, path in cfg.rgb_overlay_paths.items():
                # preprocess camera cfg
                camera_cfg = getattr(cfg.scene, camera_name)
                if "semantic_segmentation" not in camera_cfg.data_types:
                    camera_cfg.data_types.append("semantic_segmentation")
                camera_cfg.colorize_semantic_segmentation = False
                overlayed_image = self.__read_overlay_image(path, target_size=(camera_cfg.width, camera_cfg.height))
                overlayed_image = overlayed_image.to(cfg.sim.device)
                self.rgb_overlay_images[camera_name] = overlayed_image.repeat(cfg.scene.num_envs, 1, 1, 1)
                # preprocess observation cfg
                observation_cfg = getattr(cfg.observations.policy, camera_name)
                observation_cfg.func = overlay_image

        return cfg

    def __record_semantic_id_mapping(self, cfg: ManagerBasedRLDigitalTwinEnvCfg):
        for camera_name in cfg.rgb_overlay_paths.keys():
            for semantic_id, label in (
                self.scene.sensors[camera_name].data.info["semantic_segmentation"]["idToLabels"].items()
            ):
                if label["class"] == "foreground":
                    self.foreground_semantic_id_mapping[camera_name] = int(semantic_id)
                    break
