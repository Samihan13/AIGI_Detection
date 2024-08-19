# factory/model_factory.py
from models.resnet_model import ResNet50
#from models.transfer_model import TransferModel
#from models.self_supervised_model import SelfSupervisedModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, **kwargs):
        if model_type == 'resnet50':
            return ResNet50(pretrained=False, **kwargs)  # Ensure pretrained is set to False
        elif model_type == 'transfer':
            return TransferModel(**kwargs)
        elif model_type == 'self_supervised':
            return SelfSupervisedModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type {model_type}")
