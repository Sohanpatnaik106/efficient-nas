import torch
import torch.nn as nn
from functools import partial
from typing import Union, List, Dict, Any, Optional, cast

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

__all__ = [
    "VGG",
    "VGG11_Weights",
    "VGG11_BN_Weights",
    "VGG13_Weights",
    "VGG13_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights",
    "VGG19_Weights",
    "VGG19_BN_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]

class BaseModel(nn.Module):

    def __init__(self, model_name, model_config, num_classes = 100, init_weights = True, dropout = 0.5, 
                batch_norm = True, weights = None, progress = True, track_running_stats = False):
        super(BaseModel, self).__init__()

        self.model_name = model_name
        self.model_config = model_config
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = weights
        self.progress = progress
        self.track_running_stats = track_running_stats

        if "vgg" in self.model_name:
            self.model = _vgg(self.model_config, self.batch_norm, self.weights, self.progress, num_classes = self.num_classes, 
                            init_weights = self.init_weights, dropout = self.dropout, track_running_stats = self.track_running_stats)

    def forward(self, images):
        return self.model(images)

    def feature_forward(self, images):
        return self.model.feature_forward(images)

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            
            for m in self.modules():
            
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

    def feature_forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def make_layers(cfg: str, batch_norm: bool = False, track_running_stats: bool = False) -> nn.Sequential:
    
    layers: List[nn.Module] = []
    in_channels = 3
    
    for v in cfg:
    
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
    
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, track_running_stats = track_running_stats), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
    
    return nn.Sequential(*layers)

def _vgg(cfg: str, batch_norm: bool, weights, progress: bool, 
        num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5,
        track_running_stats: bool = False) -> VGG:
    
    kwargs = {}
    if weights is not None:
    
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    
    model = VGG(make_layers(cfg, batch_norm=batch_norm, track_running_stats = track_running_stats), 
                num_classes = num_classes, init_weights = init_weights, dropout = dropout)
    
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    
    return model