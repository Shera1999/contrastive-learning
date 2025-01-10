from data.data_augmentor import DataAugmentor

# Define config file paths
augmentations_config = "configs/augmentations_config.yaml"
dataset_config = "configs/dataset_config.yaml"
main_config = "configs/main_config.yaml"

# Initialize data augmentor
data_augmentor = DataAugmentor(augmentations_config, dataset_config, main_config)

# Get dataloaders
dataloader_train, dataloader_test = data_augmentor.get_dataloaders()

