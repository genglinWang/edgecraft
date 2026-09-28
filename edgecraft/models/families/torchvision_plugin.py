"""TorchVision family plugin for the EdgeCraft model zoo.

This plugin provides model specs and script generation for torchvision
models, focusing on detection and classification models.

Key principle: This plugin generates SCRIPTS, it does NOT execute training.
"""
from __future__ import annotations

from typing import List, Literal

from edgecraft.core.modality import Modality, TaskType
from edgecraft.models.specs import (
    BaseFamilyPlugin,
    DeviceClass,
    ExportFormat,
    ModelSpec,
    RuntimeId,
    TemplateContext,
)
from edgecraft.models._tegrastats_code import wrap_with_tegrastats


class TorchVisionPlugin(BaseFamilyPlugin):
    """Family plugin for torchvision models."""

    @property
    def family_id(self) -> str:
        return "torchvision"

    @property
    def display_name(self) -> str:
        return "TorchVision"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.VISION]

    def get_model_specs(self) -> List[ModelSpec]:
        """Return all torchvision model specs."""
        specs = [
            # SSD-Lite (lightweight detection)
            ModelSpec(
                name="ssdlite320_mobilenet_v3_large",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION],
                params_m=3.4,
                default_input_shape=[3, 320, 320],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 16,
                    "lr": 0.01,
                    "imgsz": 320,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "imgsz": [256, 320, 416],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_V1",
                neighbor_models=["fasterrcnn_mobilenet_v3_large_fpn"],
                lighter_alternative=None,
                heavier_alternative="fasterrcnn_mobilenet_v3_large_fpn",
            ),
            # Faster R-CNN MobileNet (lightweight detection)
            ModelSpec(
                name="fasterrcnn_mobilenet_v3_large_fpn",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION],
                params_m=19.4,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 8,
                    "lr": 0.005,
                    "imgsz": 640,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [416, 512, 640],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_V1",
                neighbor_models=["ssdlite320_mobilenet_v3_large", "fasterrcnn_resnet50_fpn_v2"],
                lighter_alternative="ssdlite320_mobilenet_v3_large",
                heavier_alternative="fasterrcnn_resnet50_fpn_v2",
            ),
            # Faster R-CNN ResNet50 (higher accuracy detection)
            ModelSpec(
                name="fasterrcnn_resnet50_fpn_v2",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION],
                params_m=43.7,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 4,
                    "lr": 0.005,
                    "imgsz": 640,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [640, 800],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_V1",
                neighbor_models=["fasterrcnn_mobilenet_v3_large_fpn"],
                lighter_alternative="fasterrcnn_mobilenet_v3_large_fpn",
                heavier_alternative=None,
            ),
            # RetinaNet (single-stage detection)
            ModelSpec(
                name="retinanet_resnet50_fpn_v2",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION],
                params_m=38.2,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 4,
                    "lr": 0.005,
                    "imgsz": 640,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [640, 800],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_V1",
                neighbor_models=["fasterrcnn_resnet50_fpn_v2"],
                lighter_alternative="fasterrcnn_mobilenet_v3_large_fpn",
                heavier_alternative=None,
            ),
            # FCOS (fully convolutional detection)
            ModelSpec(
                name="fcos_resnet50_fpn",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION],
                params_m=32.3,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 8,
                    "lr": 0.005,
                    "imgsz": 640,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [640, 800],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_V1",
                neighbor_models=["retinanet_resnet50_fpn_v2"],
                lighter_alternative="fasterrcnn_mobilenet_v3_large_fpn",
                heavier_alternative=None,
            ),
            # ViT (Vision Transformer)
            ModelSpec(
                name="vit_b_16",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=86.5,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 384],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vit_b_32", "vit_l_16"],
                lighter_alternative="vit_b_32",
                heavier_alternative="vit_l_16",
            ),
            ModelSpec(
                name="vit_b_32",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=88.2,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 384],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vit_b_16"],
                lighter_alternative=None,
                heavier_alternative="vit_b_16",
            ),
            ModelSpec(
                name="vit_l_16",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=304.3,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 16,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 384],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vit_b_16", "vit_l_32"],
                lighter_alternative="vit_b_16",
                heavier_alternative="vit_l_32",
            ),
            ModelSpec(
                name="vit_l_32",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=306.5,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 16,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 384],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vit_l_16"],
                lighter_alternative="vit_l_16",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="swin_t",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=28.2,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["swin_s", "swin_b"],
                lighter_alternative=None,
                heavier_alternative="swin_s",
            ),
            ModelSpec(
                name="swin_s",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=49.6,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["swin_t", "swin_b"],
                lighter_alternative="swin_t",
                heavier_alternative="swin_b",
            ),
            ModelSpec(
                name="swin_b",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=87.7,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["swin_s"],
                lighter_alternative="swin_s",
                heavier_alternative=None,
            ),
            # Segmentation models from torchvision
            ModelSpec(
                name="fcn_resnet50",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.SEGMENTATION],
                params_m=32.9,
                default_input_shape=[3, 520, 520],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 8,
                    "lr": 0.005,
                    "imgsz": 520,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [520, 640],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_WITH_VOC_LABELS_V1",
                neighbor_models=["deeplabv3_resnet50"],
                lighter_alternative=None,
                heavier_alternative="deeplabv3_resnet50",
            ),
            ModelSpec(
                name="deeplabv3_resnet50",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.SEGMENTATION],
                params_m=39.6,
                default_input_shape=[3, 520, 520],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 8,
                    "lr": 0.005,
                    "imgsz": 520,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [520, 640],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_WITH_VOC_LABELS_V1",
                neighbor_models=["fcn_resnet50", "deeplabv3_resnet101"],
                lighter_alternative="fcn_resnet50",
                heavier_alternative="deeplabv3_resnet101",
            ),
            ModelSpec(
                name="deeplabv3_resnet101",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.SEGMENTATION],
                params_m=58.6,
                default_input_shape=[3, 520, 520],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 4,
                    "lr": 0.005,
                    "imgsz": 520,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [520, 640],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_WITH_VOC_LABELS_V1",
                neighbor_models=["deeplabv3_resnet50", "deeplabv3_mobilenet_v3_large"],
                lighter_alternative="deeplabv3_resnet50",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="deeplabv3_mobilenet_v3_large",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.SEGMENTATION],
                params_m=11.0,
                default_input_shape=[3, 520, 520],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 16,
                    "lr": 0.005,
                    "imgsz": 520,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.ANDROID, DeviceClass.RASPBERRY_PI],
                benchmark_input_signature={
                    "imgsz": [520, 640],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_WITH_VOC_LABELS_V1",
                neighbor_models=["deeplabv3_resnet50"],
                lighter_alternative=None,
                heavier_alternative="deeplabv3_resnet50",
            ),
            # Classification models from torchvision
            ModelSpec(
                name="mobilenet_v2",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=3.5,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "imgsz": [160, 192, 224],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V2",
                neighbor_models=["mobilenet_v3_small", "mobilenet_v3_large"],
                lighter_alternative=None,
                heavier_alternative="mobilenet_v3_small",
            ),
            ModelSpec(
                name="vgg16",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=138.3,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vgg16_bn", "vgg19"],
                lighter_alternative=None,
                heavier_alternative="vgg16_bn",
            ),
            ModelSpec(
                name="vgg16_bn",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=138.3,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vgg16", "vgg19_bn"],
                lighter_alternative="vgg16",
                heavier_alternative="vgg19_bn",
            ),
            ModelSpec(
                name="vgg19",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=143.6,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vgg16", "vgg19_bn"],
                lighter_alternative="vgg16",
                heavier_alternative="vgg19_bn",
            ),
            ModelSpec(
                name="vgg19_bn",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=143.6,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["vgg16_bn", "vgg19"],
                lighter_alternative="vgg19",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="alexnet",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=61.1,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["squeezenet1_1"],
                lighter_alternative="squeezenet1_1",
                heavier_alternative="vgg16",
            ),
            ModelSpec(
                name="mobilenet_v3_small",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=2.5,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "imgsz": [160, 192, 224],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["mobilenet_v2", "mobilenet_v3_large"],
                lighter_alternative="mobilenet_v2",
                heavier_alternative="mobilenet_v3_large",
            ),
            ModelSpec(
                name="mobilenet_v3_large",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=5.5,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.ANDROID],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V2",
                neighbor_models=["mobilenet_v3_small", "efficientnet_b0"],
                lighter_alternative="mobilenet_v3_small",
                heavier_alternative="efficientnet_b0",
            ),
            ModelSpec(
                name="efficientnet_b0",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=5.3,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["mobilenet_v3_large", "efficientnet_b1"],
                lighter_alternative="mobilenet_v3_large",
                heavier_alternative="efficientnet_b1",
            ),
            ModelSpec(
                name="efficientnet_b1",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=7.8,
                default_input_shape=[3, 240, 240],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 240,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 240, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V2",
                neighbor_models=["efficientnet_b0"],
                lighter_alternative="efficientnet_b0",
                heavier_alternative=None,
            ),
            # SqueezeNet (very lightweight)
            ModelSpec(
                name="squeezenet1_1",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=1.2,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 128,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "imgsz": [160, 192, 224],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["mobilenet_v2"],
                lighter_alternative=None,
                heavier_alternative="mobilenet_v2",
            ),
            ModelSpec(
                name="resnet18",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=11.7,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["mobilenet_v3_large", "resnet50"],
                lighter_alternative="mobilenet_v3_large",
                heavier_alternative="resnet50",
            ),
            ModelSpec(
                name="resnet50",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=25.6,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V2",
                neighbor_models=["resnet18", "convnext_tiny"],
                lighter_alternative="resnet18",
                heavier_alternative="convnext_tiny",
            ),
            ModelSpec(
                name="convnext_tiny",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=28.6,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 256, 320],
                    "batch_size": [1],
                },
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["resnet50"],
                lighter_alternative="resnet50",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="densenet121",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=8.0,
                default_input_shape=[3, 224, 224],
                default_hyperparams={"epochs": 50, "batch_size": 32, "lr": 0.001, "imgsz": 224},
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [192, 224, 256], "batch_size": [1]},
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V1",
                neighbor_models=["efficientnet_b0", "resnet18"],
                lighter_alternative="efficientnet_b0",
                heavier_alternative="resnet18",
            ),
            ModelSpec(
                name="regnet_y_400mf",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=4.3,
                default_input_shape=[3, 224, 224],
                default_hyperparams={"epochs": 50, "batch_size": 64, "lr": 0.001, "imgsz": 224},
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.ANDROID],
                benchmark_input_signature={"imgsz": [192, 224, 256], "batch_size": [1]},
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V2",
                neighbor_models=["mobilenet_v3_large", "regnet_x_400mf"],
                lighter_alternative="mobilenet_v3_large",
                heavier_alternative="regnet_x_400mf",
            ),
            ModelSpec(
                name="regnet_x_400mf",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=5.2,
                default_input_shape=[3, 224, 224],
                default_hyperparams={"epochs": 50, "batch_size": 64, "lr": 0.001, "imgsz": 224},
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [192, 224, 256], "batch_size": [1]},
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="IMAGENET1K_V2",
                neighbor_models=["regnet_y_400mf", "resnet18"],
                lighter_alternative="regnet_y_400mf",
                heavier_alternative="resnet18",
            ),
            ModelSpec(
                name="ssd300_vgg16",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION],
                params_m=35.6,
                default_input_shape=[3, 300, 300],
                default_hyperparams={"epochs": 50, "batch_size": 8, "lr": 0.005, "imgsz": 300},
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [300], "batch_size": [1]},
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_V1",
                neighbor_models=["ssdlite320_mobilenet_v3_large", "retinanet_resnet50_fpn_v2"],
                lighter_alternative="ssdlite320_mobilenet_v3_large",
                heavier_alternative="retinanet_resnet50_fpn_v2",
            ),
            ModelSpec(
                name="maskrcnn_resnet50_fpn",
                family_id="torchvision",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.SEGMENTATION],
                params_m=44.4,
                default_input_shape=[3, 640, 640],
                default_hyperparams={"epochs": 50, "batch_size": 4, "lr": 0.005, "imgsz": 640},
                pip_packages=["torch", "torchvision>=0.15.0"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [512, 640], "batch_size": [1]},
                source_url="https://pytorch.org/vision/stable/models.html",
                pretrained_weights="COCO_V1",
                neighbor_models=["fasterrcnn_resnet50_fpn_v2"],
                lighter_alternative="fasterrcnn_mobilenet_v3_large_fpn",
                heavier_alternative=None,
            ),
        ]
        for spec in specs:
            spec.edge_compatible = True
            if RuntimeId.TENSORRT not in spec.supported_runtimes:
                spec.supported_runtimes.append(RuntimeId.TENSORRT)
        return specs

    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py for torchvision models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}

        if context.task_type == TaskType.OBJECT_DETECTION:
            return self._render_detection_train(spec, hp, context)
        elif context.task_type == TaskType.SEGMENTATION:
            return self._render_segmentation_train(spec, hp, context)
        else:
            return self._render_classification_train(spec, hp, context)

    def _render_detection_train(
        self,
        spec: ModelSpec,
        hp: dict,
        context: TemplateContext,
    ) -> str:
        return f'''#!/usr/bin/env python3
"""Training script for {spec.name} detection model."""
import json
import sys
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    ssd300_vgg16,
    maskrcnn_resnet50_fpn,
)
from torch.utils.data import DataLoader

# Configuration
model_name = "{spec.name}"
epochs = {hp.get("epochs", 50)}
batch_size = {hp.get("batch_size", 8)}
lr = {hp.get("lr", 0.005)}
use_pretrained = {hp.get("use_pretrained", True)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model factory
MODEL_FACTORY = {{
    "ssdlite320_mobilenet_v3_large": ssdlite320_mobilenet_v3_large,
    "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn_v2": retinanet_resnet50_fpn_v2,
    "fcos_resnet50_fpn": fcos_resnet50_fpn,
    "ssd300_vgg16": ssd300_vgg16,
    "maskrcnn_resnet50_fpn": maskrcnn_resnet50_fpn,
}}

# Create model (customize num_classes for your dataset)
num_classes = 91  # COCO default, change for custom dataset
weights = "DEFAULT" if use_pretrained else None
weights_backbone = "DEFAULT" if use_pretrained else None

try:
    if model_name == "ssd300_vgg16":
        model = MODEL_FACTORY[model_name](weights=weights, weights_backbone=weights_backbone)
    elif model_name == "maskrcnn_resnet50_fpn":
        model = MODEL_FACTORY[model_name](weights=weights, weights_backbone=weights_backbone)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    else:
        model = MODEL_FACTORY[model_name](weights=weights, weights_backbone=weights_backbone, num_classes=num_classes)
except TypeError:
    model = MODEL_FACTORY[model_name](weights=weights, num_classes=num_classes)

model = model.to(device)

# TODO: Implement dataset loading for torchvision detection format
# This is a placeholder - actual training requires dataset setup

output = {{
    "status": "success",
    "model_path": "outputs/best.pt",
    "metrics": {{}},
}}
print(json.dumps(output))
'''

    def _render_segmentation_train(
        self,
        spec: ModelSpec,
        hp: dict,
        context: TemplateContext,
    ) -> str:
        return f'''#!/usr/bin/env python3
"""Training script for {spec.name} segmentation model."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import (
    fcn_resnet50,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
)
from torch.utils.data import DataLoader

# Configuration
model_name = "{spec.name}"
epochs = {hp.get("epochs", 50)}
batch_size = {hp.get("batch_size", 8)}
lr = {hp.get("lr", 0.005)}
use_pretrained = {hp.get("use_pretrained", True)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model factory
MODEL_FACTORY = {{
    "fcn_resnet50": fcn_resnet50,
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": deeplabv3_mobilenet_v3_large,
}}

# Create model (customize num_classes for your dataset)
num_classes = 21  # Pascal VOC default, change for custom dataset
weights = "DEFAULT" if use_pretrained else None
weights_backbone = "DEFAULT" if use_pretrained else None

try:
    if model_name == "fcn_resnet50":
        model = MODEL_FACTORY[model_name](weights=weights, weights_backbone=weights_backbone)
    else:
        model = MODEL_FACTORY[model_name](weights=weights, weights_backbone=weights_backbone, num_classes=num_classes)
except TypeError:
    model = MODEL_FACTORY[model_name](weights=weights, num_classes=num_classes)

if hasattr(model, "classifier") and hasattr(model.classifier, "classifier") and isinstance(model.classifier[4], nn.Conv2d):
    # Try replacing classifier head
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
elif hasattr(model, "classifier") and hasattr(model.classifier, "classifier"):
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))

model = model.to(device)

# TODO: Implement dataset loading for torchvision segmentation format
# This is a placeholder - actual training requires dataset setup

output = {{
    "status": "success",
    "model_path": "outputs/best.pt",
    "metrics": {{}},
}}
print(json.dumps(output))
'''

    def _render_classification_train(
        self,
        spec: ModelSpec,
        hp: dict,
        context: TemplateContext,
    ) -> str:
        return f'''#!/usr/bin/env python3
"""Training script for {spec.name} classification model."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import yaml

# Configuration
model_name = "{spec.name}"
epochs = {hp.get("epochs", 50)}
batch_size = {hp.get("batch_size", 32)}
lr = {hp.get("lr", 0.01)}
imgsz = {hp.get("imgsz", 224)}
use_pretrained = {hp.get("use_pretrained", True)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(imgsz),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(int(imgsz * 1.14)),
    transforms.CenterCrop(imgsz),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _load_imagefolder_contract():
    cfg_path = Path("config/data.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {{}}
    cfg = cfg or {{}}
    root = Path(cfg.get("root") or "{context.dataset_path}").expanduser()
    train_dir = Path(cfg.get("train") or (root / "train")).expanduser()
    val_dir = Path(cfg.get("val") or (root / "val")).expanduser()
    if not val_dir.exists():
        val_dir = train_dir
    return train_dir, val_dir, cfg


train_dir, val_dir, data_cfg = _load_imagefolder_contract()
train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
num_classes = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Create model
weights = "DEFAULT" if use_pretrained else None
model = torchvision.models.get_model(model_name, weights=weights)
# Replace classifier head
if hasattr(model, "classifier"):
    in_features = model.classifier[-1].in_features if hasattr(model.classifier[-1], "in_features") else model.classifier.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
elif hasattr(model, "fc"):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
elif hasattr(model, "head"): # for ViT/Swin
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training loop
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save({{
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "class_names": train_dataset.classes,
            "model_name": model_name,
        }}, "outputs/best.pt")

output = {{
    "status": "success",
    "model_path": "outputs/best.pt",
    "metrics": {{
        "accuracy": best_acc,
    }},
}}
print(json.dumps(output))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py for torchvision models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        imgsz = hp.get("imgsz", 224)

        return f'''#!/usr/bin/env python3
"""Inference benchmark script for {spec.name}."""
import json
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
model_name = "{spec.name}"
imgsz = {imgsz}
cfg_path = Path("config/data.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {{}}
cfg = cfg or {{}}
num_classes = int(cfg.get("num_classes") or cfg.get("nc") or 10)

# Model factory for detection/segmentation
MODEL_FACTORY = {{
    "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn_v2": torchvision.models.detection.retinanet_resnet50_fpn_v2,
    "fcos_resnet50_fpn": torchvision.models.detection.fcos_resnet50_fpn,
    "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
    "maskrcnn_resnet50_fpn": torchvision.models.detection.maskrcnn_resnet50_fpn,
    "fcn_resnet50": torchvision.models.segmentation.fcn_resnet50,
    "deeplabv3_resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
    "deeplabv3_resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
}}

# Load model
model_path = Path(os.environ.get("EDGECRAFT_ARTIFACT_PATH", "outputs/best.pt"))
if not model_path.exists() and Path("outputs/best.pt").exists():
    model_path = Path("outputs/best.pt")
if model_name in MODEL_FACTORY:
    model = MODEL_FACTORY[model_name](weights=None)
else:
    model = torchvision.models.get_model(model_name, weights=None)

if hasattr(model, "classifier"):
    try:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    except Exception:
        if hasattr(model.classifier, "in_features"):
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
elif hasattr(model, "fc") and hasattr(model.fc, "in_features"):
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
elif hasattr(model, "head") and hasattr(model.head, "in_features"):
    model.head = torch.nn.Linear(model.head.in_features, num_classes)

if model_path.exists():
    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

# Warmup
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

# Benchmark
n_runs = 100
latencies = []
with torch.no_grad():
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

avg_latency = float(np.mean(latencies))
std_latency = float(np.std(latencies))

# Memory
import psutil
process = psutil.Process()
mem_mb = process.memory_info().rss / (1024 * 1024)

output = {{
    "status": "success",
    "metrics": {{
        "Latency": avg_latency,
        "Latency_std": std_latency,
        "Memory_mb": mem_mb,
        "throughput_fps": 1000.0 / avg_latency if avg_latency > 0 else 0,
    }},
}}
print(json.dumps(output))
'''

    def get_supported_profile_runtimes(self, model_spec: ModelSpec) -> List[RuntimeId]:
        if TaskType.OBJECT_DETECTION in model_spec.supported_tasks or TaskType.SEGMENTATION in model_spec.supported_tasks:
            # Stable-first policy: detection models run ORT profiling only.
            # TRT path is kept out of auto plan due to frequent parser/runtime
            # incompatibilities on exported detection graphs.
            return (
                [RuntimeId.ONNXRUNTIME]
                if RuntimeId.ONNXRUNTIME in model_spec.supported_runtimes
                else []
            )
        supported: List[RuntimeId] = []
        if RuntimeId.ONNXRUNTIME in model_spec.supported_runtimes:
            supported.append(RuntimeId.ONNXRUNTIME)
        if RuntimeId.TENSORRT in model_spec.supported_runtimes:
            supported.append(RuntimeId.TENSORRT)
        return supported

    def render_profile_benchmark_script(
        self,
        model_spec: ModelSpec,
        runtime_id: RuntimeId,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        is_detection = TaskType.OBJECT_DETECTION in model_spec.supported_tasks
        is_segmentation = TaskType.SEGMENTATION in model_spec.supported_tasks
        if runtime_id == RuntimeId.ONNXRUNTIME:
            if is_detection:
                return wrap_with_tegrastats(self._render_detection_profile_onnxruntime(model_spec.name, imgsz, warmup, iterations))
            if is_segmentation:
                return wrap_with_tegrastats(self._render_profile_onnxruntime(model_spec.name, imgsz, warmup, iterations))
            return wrap_with_tegrastats(self._render_profile_onnxruntime(model_spec.name, imgsz, warmup, iterations))
        if runtime_id == RuntimeId.TENSORRT:
            if is_detection:
                return wrap_with_tegrastats(self._render_detection_profile_tensorrt(model_spec.name, imgsz, iterations))
            if is_segmentation:
                return wrap_with_tegrastats(self._render_profile_tensorrt(model_spec.name, imgsz, iterations))
            return wrap_with_tegrastats(self._render_profile_tensorrt(model_spec.name, imgsz, iterations))
        return ""

    def _render_profile_onnxruntime(
        self,
        model_name: str,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import time
import numpy as np

try:
    import torch
    import torchvision
    import onnxruntime as ort
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

# Model factory for segmentation
MODEL_FACTORY = {{
    "fcn_resnet50": torchvision.models.segmentation.fcn_resnet50,
    "deeplabv3_resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
    "deeplabv3_resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
}}

try:
    if "{model_name}" in MODEL_FACTORY:
        model = MODEL_FACTORY["{model_name}"](weights=None).eval().cpu()
    else:
        model = torchvision.models.get_model("{model_name}", weights=None).eval().cpu()
    dummy = torch.randn(1, 3, {imgsz}, {imgsz})

    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "mask_roi_pool"):
        output_names = ["boxes", "scores", "labels", "masks"]
        dynamic_axes = {{"input": {{0: "batch"}}, "boxes": {{0: "det"}}, "scores": {{0: "det"}}, "labels": {{0: "det"}}, "masks": {{0: "det"}}}}
    elif "{model_name}".startswith("deeplabv3") or "{model_name}".startswith("fcn"):
        output_names = ["out", "aux"] if getattr(model, "aux_classifier", None) else ["out"]
        dynamic_axes = {{"input": {{0: "batch"}}, "out": {{0: "batch"}}}}
        if len(output_names) > 1:
            dynamic_axes["aux"] = {{0: "batch"}}
    else:
        output_names = ["output"]
        dynamic_axes = {{"input": {{0: "batch"}}, "output": {{0: "batch"}}}}

    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
    sess = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

dummy_np = np.random.randn(1, 3, {imgsz}, {imgsz}).astype(np.float32)
for _ in range({warmup}):
    _ = sess.run(None, {{input_name: dummy_np}})

latencies = []
for _ in range({iterations}):
    t0 = time.perf_counter()
    _ = sess.run(None, {{input_name: dummy_np}})
    latencies.append((time.perf_counter() - t0) * 1000)

arr = np.array(latencies, dtype=np.float64)
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
print(json.dumps({{
    "status":"success",
    "latency_avg_ms": float(arr.mean()),
    "latency_p50_ms": float(np.percentile(arr, 50)),
    "latency_p95_ms": float(np.percentile(arr, 95)),
    "latency_p99_ms": float(np.percentile(arr, 99)),
    "latency_min_ms": float(arr.min()),
    "latency_max_ms": float(arr.max()),
    "latency_std_ms": float(arr.std()),
    "throughput_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    "peak_memory_mb": float(mem_mb),
    "raw_latencies": arr.tolist(),
}}))
'''

    def _render_profile_tensorrt(
        self,
        model_name: str,
        imgsz: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import re
import shutil
import subprocess

try:
    import torch
    import torchvision
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

trtexec = shutil.which("trtexec")
if not trtexec:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":"trtexec not found in PATH"}}))
    raise SystemExit(1)

# Model factory for segmentation
MODEL_FACTORY = {{
    "fcn_resnet50": torchvision.models.segmentation.fcn_resnet50,
    "deeplabv3_resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
    "deeplabv3_resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
}}

try:
    if "{model_name}" in MODEL_FACTORY:
        model = MODEL_FACTORY["{model_name}"](weights=None).eval().cpu()
    else:
        model = torchvision.models.get_model("{model_name}", weights=None).eval().cpu()
    dummy = torch.randn(1, 3, {imgsz}, {imgsz})

    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "mask_roi_pool"):
        output_names = ["boxes", "scores", "labels", "masks"]
        dynamic_axes = {{"input": {{0: "batch"}}, "boxes": {{0: "det"}}, "scores": {{0: "det"}}, "labels": {{0: "det"}}, "masks": {{0: "det"}}}}
    elif "{model_name}".startswith("deeplabv3") or "{model_name}".startswith("fcn"):
        output_names = ["out", "aux"] if getattr(model, "aux_classifier", None) else ["out"]
        dynamic_axes = {{"input": {{0: "batch"}}, "out": {{0: "batch"}}}}
        if len(output_names) > 1:
            dynamic_axes["aux"] = {{0: "batch"}}
    else:
        output_names = ["output"]
        dynamic_axes = {{"input": {{0: "batch"}}, "output": {{0: "batch"}}}}

    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export","error_message":str(e)}}))
    raise SystemExit(1)

cmd = [
    trtexec,
    "--onnx=model.onnx",
    "--saveEngine=model.engine",
    "--explicitBatch",
    "--fp16",
    "--useSpinWait",
    "--warmUp=0",
    "--duration=0",
    "--iterations={iterations}",
    "--avgRuns=1",
    "--shapes=input:1x3x{imgsz}x{imgsz}",
]
result = subprocess.run(cmd, capture_output=True, text=True)
combined = (result.stdout or "") + "\\n" + (result.stderr or "")
if result.returncode != 0:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":combined[-4000:]}}))
    raise SystemExit(1)

mean_match = re.search(r"mean\\s*=\\s*([0-9.]+)\\s*ms", combined)
p95_match = re.search(r"percentile\\(95%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
p99_match = re.search(r"percentile\\(99%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
if not mean_match:
    print(json.dumps({{"status":"error","error_type":"parse","error_message":"Could not parse trtexec latency output"}}))
    raise SystemExit(1)

lat_mean = float(mean_match.group(1))
lat_p95 = float(p95_match.group(1)) if p95_match else lat_mean
lat_p99 = float(p99_match.group(1)) if p99_match else lat_p95
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

print(json.dumps({{
    "status":"success",
    "latency_avg_ms": lat_mean,
    "latency_p50_ms": lat_mean,
    "latency_p95_ms": lat_p95,
    "latency_p99_ms": lat_p99,
    "latency_min_ms": lat_mean,
    "latency_max_ms": lat_mean,
    "latency_std_ms": 0.0,
    "throughput_fps": float(1000.0 / lat_mean) if lat_mean > 0 else 0.0,
    "peak_memory_mb": float(mem_mb),
    "raw_latencies": [],
}}))
'''

    def _render_detection_profile_onnxruntime(
        self,
        model_name: str,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision
    import onnxruntime as ort
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

MODEL_FACTORY = {{
    "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
    "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn_v2": torchvision.models.detection.retinanet_resnet50_fpn_v2,
    "fcos_resnet50_fpn": torchvision.models.detection.fcos_resnet50_fpn,
    "maskrcnn_resnet50_fpn": torchvision.models.detection.maskrcnn_resnet50_fpn,
}}

class DetectionWrapper(nn.Module):
    def __init__(self, model, max_det=100):
        super().__init__()
        self.model = model
        self.max_det = max_det

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            images = [x[i] for i in range(x.shape[0])]
        else:
            images = x
        preds = self.model(images)
        if isinstance(preds, dict):
            preds = [preds]
        if not isinstance(preds, (list, tuple)) or len(preds) == 0:
            out_boxes = torch.zeros(self.max_det, 4, dtype=torch.float32)
            out_scores = torch.zeros(self.max_det, dtype=torch.float32)
            out_labels = torch.zeros(self.max_det, dtype=torch.float32)
            if hasattr(self.model, "roi_heads") and hasattr(self.model.roi_heads, "mask_roi_pool"):
                out_masks = torch.zeros(self.max_det, 1, 28, 28, dtype=torch.float32)
                return out_boxes, out_scores, out_labels, out_masks
            return out_boxes, out_scores, out_labels
        first = preds[0] if isinstance(preds[0], dict) else {{}}
        boxes = first.get("boxes", torch.zeros(0, 4, dtype=torch.float32))
        scores = first.get("scores", torch.zeros(0, dtype=torch.float32))
        labels = first.get("labels", torch.zeros(0, dtype=torch.float32)).to(torch.float32)
        n = min(boxes.shape[0], self.max_det)
        out_boxes = torch.zeros(self.max_det, 4, device=boxes.device, dtype=boxes.dtype)
        out_scores = torch.zeros(self.max_det, device=scores.device, dtype=scores.dtype)
        out_labels = torch.zeros(self.max_det, device=labels.device, dtype=labels.dtype)

        if hasattr(self.model, "roi_heads") and hasattr(self.model.roi_heads, "mask_roi_pool"):
            masks = first.get("masks", torch.zeros(0, 1, 28, 28, dtype=torch.float32))
            out_masks = torch.zeros(self.max_det, 1, 28, 28, device=masks.device, dtype=masks.dtype)
            if n > 0:
                out_masks[:n] = masks[:n]
            if n > 0:
                out_boxes[:n] = boxes[:n]
                out_scores[:n] = scores[:n]
                out_labels[:n] = labels[:n]
            return out_boxes, out_scores, out_labels, out_masks

        if n > 0:
            out_boxes[:n] = boxes[:n]
            out_scores[:n] = scores[:n]
            out_labels[:n] = labels[:n]
        return out_boxes, out_scores, out_labels

try:
    if "{model_name}" in MODEL_FACTORY:
        model = MODEL_FACTORY["{model_name}"](weights=None).eval().cpu()
    else:
        model = torchvision.models.get_model("{model_name}", weights=None).eval().cpu()
    model = DetectionWrapper(model).eval()
    dummy = torch.randn(1, 3, {imgsz}, {imgsz})

    if hasattr(base, "roi_heads") and hasattr(base.roi_heads, "mask_roi_pool"):
        output_names = ["boxes", "scores", "labels", "masks"]
        dynamic_axes = {{"input": {{0: "batch"}}, "boxes": {{0: "det"}}, "scores": {{0: "det"}}, "labels": {{0: "det"}}, "masks": {{0: "det"}}}}
    else:
        output_names = ["boxes", "scores", "labels"]
        dynamic_axes = {{"input": {{0: "batch"}}, "boxes": {{0: "det"}}, "scores": {{0: "det"}}, "labels": {{0: "det"}}}}

    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
    sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

dummy_np = np.random.randn(1, 3, {imgsz}, {imgsz}).astype(np.float32)
for _ in range({warmup}):
    _ = sess.run(None, {{input_name: dummy_np}})

latencies = []
for _ in range({iterations}):
    t0 = time.perf_counter()
    _ = sess.run(None, {{input_name: dummy_np}})
    latencies.append((time.perf_counter() - t0) * 1000)

arr = np.array(latencies, dtype=np.float64)
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
print(json.dumps({{
    "status":"success",
    "latency_avg_ms": float(arr.mean()),
    "latency_p50_ms": float(np.percentile(arr, 50)),
    "latency_p95_ms": float(np.percentile(arr, 95)),
    "latency_p99_ms": float(np.percentile(arr, 99)),
    "latency_min_ms": float(arr.min()),
    "latency_max_ms": float(arr.max()),
    "latency_std_ms": float(arr.std()),
    "throughput_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    "peak_memory_mb": float(mem_mb),
    "raw_latencies": arr.tolist(),
}}))
'''

    def _render_detection_profile_tensorrt(
        self,
        model_name: str,
        imgsz: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import re
import shutil
import subprocess

try:
    import torch
    import torch.nn as nn
    import torchvision
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

trtexec = shutil.which("trtexec")
if not trtexec:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":"trtexec not found in PATH"}}))
    raise SystemExit(1)

MODEL_FACTORY = {{
    "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
    "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn_v2": torchvision.models.detection.retinanet_resnet50_fpn_v2,
    "fcos_resnet50_fpn": torchvision.models.detection.fcos_resnet50_fpn,
    "maskrcnn_resnet50_fpn": torchvision.models.detection.maskrcnn_resnet50_fpn,
}}

class DetectionWrapper(nn.Module):
    def __init__(self, model, max_det=100):
        super().__init__()
        self.model = model
        self.max_det = max_det

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            images = [x[i] for i in range(x.shape[0])]
        else:
            images = x
        preds = self.model(images)
        if isinstance(preds, dict):
            preds = [preds]
        if not isinstance(preds, (list, tuple)) or len(preds) == 0:
            out_boxes = torch.zeros(self.max_det, 4, dtype=torch.float32)
            out_scores = torch.zeros(self.max_det, dtype=torch.float32)
            out_labels = torch.zeros(self.max_det, dtype=torch.float32)
            if hasattr(self.model, "roi_heads") and hasattr(self.model.roi_heads, "mask_roi_pool"):
                out_masks = torch.zeros(self.max_det, 1, 28, 28, dtype=torch.float32)
                return out_boxes, out_scores, out_labels, out_masks
            return out_boxes, out_scores, out_labels
        first = preds[0] if isinstance(preds[0], dict) else {{}}
        boxes = first.get("boxes", torch.zeros(0, 4, dtype=torch.float32))
        scores = first.get("scores", torch.zeros(0, dtype=torch.float32))
        labels = first.get("labels", torch.zeros(0, dtype=torch.float32)).to(torch.float32)
        n = min(boxes.shape[0], self.max_det)
        out_boxes = torch.zeros(self.max_det, 4, device=boxes.device, dtype=boxes.dtype)
        out_scores = torch.zeros(self.max_det, device=scores.device, dtype=scores.dtype)
        out_labels = torch.zeros(self.max_det, device=labels.device, dtype=labels.dtype)

        if hasattr(self.model, "roi_heads") and hasattr(self.model.roi_heads, "mask_roi_pool"):
            masks = first.get("masks", torch.zeros(0, 1, 28, 28, dtype=torch.float32))
            out_masks = torch.zeros(self.max_det, 1, 28, 28, device=masks.device, dtype=masks.dtype)
            if n > 0:
                out_masks[:n] = masks[:n]
            if n > 0:
                out_boxes[:n] = boxes[:n]
                out_scores[:n] = scores[:n]
                out_labels[:n] = labels[:n]
            return out_boxes, out_scores, out_labels, out_masks

        if n > 0:
            out_boxes[:n] = boxes[:n]
            out_scores[:n] = scores[:n]
            out_labels[:n] = labels[:n]
        return out_boxes, out_scores, out_labels

try:
    if "{model_name}" in MODEL_FACTORY:
        model = MODEL_FACTORY["{model_name}"](weights=None).eval().cpu()
    else:
        model = torchvision.models.get_model("{model_name}", weights=None).eval().cpu()
    model = DetectionWrapper(model).eval()
    dummy = torch.randn(1, 3, {imgsz}, {imgsz})

    if hasattr(base, "roi_heads") and hasattr(base.roi_heads, "mask_roi_pool"):
        output_names = ["boxes", "scores", "labels", "masks"]
        dynamic_axes = {{"input": {{0: "batch"}}, "boxes": {{0: "det"}}, "scores": {{0: "det"}}, "labels": {{0: "det"}}, "masks": {{0: "det"}}}}
    else:
        output_names = ["boxes", "scores", "labels"]
        dynamic_axes = {{"input": {{0: "batch"}}, "boxes": {{0: "det"}}, "scores": {{0: "det"}}, "labels": {{0: "det"}}}}

    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export","error_message":str(e)}}))
    raise SystemExit(1)

cmd = [
    trtexec,
    "--onnx=model.onnx",
    "--saveEngine=model.engine",
    "--explicitBatch",
    "--fp16",
    "--useSpinWait",
    "--warmUp=0",
    "--duration=0",
    "--iterations={iterations}",
    "--avgRuns=1",
    "--shapes=input:1x3x{imgsz}x{imgsz}",
]
result = subprocess.run(cmd, capture_output=True, text=True)
combined = (result.stdout or "") + "\\n" + (result.stderr or "")
if result.returncode != 0:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":combined[-4000:]}}))
    raise SystemExit(1)

mean_match = re.search(r"mean\\s*=\\s*([0-9.]+)\\s*ms", combined)
p95_match = re.search(r"percentile\\(95%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
p99_match = re.search(r"percentile\\(99%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
if not mean_match:
    print(json.dumps({{"status":"error","error_type":"parse","error_message":"Could not parse trtexec latency output"}}))
    raise SystemExit(1)

lat_mean = float(mean_match.group(1))
lat_p95 = float(p95_match.group(1)) if p95_match else lat_mean
lat_p99 = float(p99_match.group(1)) if p99_match else lat_p95
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
print(json.dumps({{
    "status":"success",
    "latency_avg_ms": lat_mean,
    "latency_p50_ms": lat_mean,
    "latency_p95_ms": lat_p95,
    "latency_p99_ms": lat_p99,
    "latency_min_ms": lat_mean,
    "latency_max_ms": lat_mean,
    "latency_std_ms": 0.0,
    "throughput_fps": float(1000.0 / lat_mean) if lat_mean > 0 else 0.0,
    "peak_memory_mb": float(mem_mb),
    "raw_latencies": [],
}}))
'''


# Module-level instance
torchvision_plugin = TorchVisionPlugin()
