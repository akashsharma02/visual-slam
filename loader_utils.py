from vslam.dataloader import *
from vslam.tum_dataloader import TumDataloader
from vslam.kitti_dataloader import KittiDataloader


# def create_dataloader(dataset_type: str, dataset_path: str, dataset_sequence: str) -> Dataloader:
#     """ Creates a dataloader of a given type

#     Args:
#         dataset_type: {tum, kitti, euroc}
#         dataset_path: path to the dataset
#         dataset_sequence: sequence number of dataset
#     :returns:
#         Appropriate instance of dataloader

#     """
#     if dataset_type == "tum":
#         return TumDataloader(dataset_path, dataset_sequence)
#     elif dataset_type == "kitti":
#         return KittiDataloader(dataset_path, dataset_sequence)
