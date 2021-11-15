from .blender_loader import BlenderDataset
from .llff_loader import LLFFDataset

dataset_dict = {
    'blender': BlenderDataset,
    'llff': LLFFDataset
}
