"""Timm family plugin for the EdgeCraft model zoo.

This plugin provides model specs and script generation for the timm
(PyTorch Image Models) ecosystem, focusing on edge-compatible classification
and feature extraction models.

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
    QuantMode,
    RuntimeId,
    TemplateContext,
)
from edgecraft.models._tegrastats_code import wrap_with_tegrastats


class TimmPlugin(BaseFamilyPlugin):
    """Family plugin for timm models."""

    @property
    def family_id(self) -> str:
        return "timm"

    @property
    def display_name(self) -> str:
        return "Timm (PyTorch Image Models)"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.VISION]

    def get_model_specs(self) -> List[ModelSpec]:
        """Return all timm model specs."""
        specs = [
            # MobileNetV3
            ModelSpec(
                name="mobilenetv3_small_100",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=2.5,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.RASPBERRY_PI,
                    DeviceClass.ANDROID,
                ],
                benchmark_input_signature={
                    "imgsz": [160, 192, 224],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="mobilenetv3_small_100.lamb_in1k",
                neighbor_models=["mobilenetv3_large_100"],
                lighter_alternative=None,
                heavier_alternative="mobilenetv3_large_100",
            ),
            ModelSpec(
                name="mobilenetv3_large_100",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=5.4,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="mobilenetv3_large_100.ra_in1k",
                neighbor_models=["mobilenetv3_small_100", "efficientnet_b0"],
                lighter_alternative="mobilenetv3_small_100",
                heavier_alternative="efficientnet_b0",
            ),
            # EfficientNet
            ModelSpec(
                name="efficientnet_b0",
                family_id="timm",
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
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="efficientnet_b0.ra_in1k",
                neighbor_models=["mobilenetv3_large_100", "efficientnet_b1"],
                lighter_alternative="mobilenetv3_large_100",
                heavier_alternative="efficientnet_b1",
            ),
            ModelSpec(
                name="efficientnet_b1",
                family_id="timm",
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
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 240, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="efficientnet_b1.ra_in1k",
                neighbor_models=["efficientnet_b0", "efficientnet_b2"],
                lighter_alternative="efficientnet_b0",
                heavier_alternative="efficientnet_b2",
            ),
            ModelSpec(
                name="efficientnet_b2",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=9.1,
                default_input_shape=[3, 260, 260],
                default_hyperparams={
                    "epochs": 100,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 260,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [240, 260, 288],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="efficientnet_b2.ra_in1k",
                neighbor_models=["efficientnet_b1", "efficientnet_b3"],
                lighter_alternative="efficientnet_b1",
                heavier_alternative="efficientnet_b3",
            ),
            ModelSpec(
                name="efficientnet_b3",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=12.0,
                default_input_shape=[3, 300, 300],
                default_hyperparams={
                    "epochs": 100,
                    "batch_size": 24,
                    "lr": 0.001,
                    "imgsz": 300,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [260, 300, 320],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="efficientnet_b3.ra_in1k",
                neighbor_models=["efficientnet_b2"],
                lighter_alternative="efficientnet_b2",
                heavier_alternative=None,
            ),
            # ResNet
            ModelSpec(
                name="resnet18",
                family_id="timm",
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
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="resnet18.a1_in1k",
                neighbor_models=["resnet34", "resnet50"],
                lighter_alternative=None,
                heavier_alternative="resnet34",
            ),
            ModelSpec(
                name="resnet34",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=21.8,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="resnet34.a1_in1k",
                neighbor_models=["resnet18", "resnet50"],
                lighter_alternative="resnet18",
                heavier_alternative="resnet50",
            ),
            ModelSpec(
                name="resnet50",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=25.6,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 100,
                    "batch_size": 32,
                    "lr": 0.01,
                    "imgsz": 224,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="resnet50.a1_in1k",
                neighbor_models=["resnet34"],
                lighter_alternative="resnet34",
                heavier_alternative=None,
            ),
            # EfficientViT (edge-optimized vision transformer)
            ModelSpec(
                name="efficientvit_b0",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=3.4,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="efficientvit_b0.r224_in1k",
                neighbor_models=["efficientvit_b1", "mobilenetv3_small_100"],
                lighter_alternative="mobilenetv3_small_100",
                heavier_alternative="efficientvit_b1",
            ),
            ModelSpec(
                name="efficientvit_b1",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=9.1,
                default_input_shape=[3, 256, 256],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 256,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [224, 256, 288],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="efficientvit_b1.r256_in1k",
                neighbor_models=["efficientvit_b0", "efficientvit_b2"],
                lighter_alternative="efficientvit_b0",
                heavier_alternative="efficientvit_b2",
            ),
            # MobileViT
            ModelSpec(
                name="mobilevit_xxs",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=1.3,
                default_input_shape=[3, 256, 256],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 256,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="mobilevit_xxs.cvnets_in1k",
                neighbor_models=["mobilevit_xs", "mobilenetv3_small_100"],
                lighter_alternative=None,
                heavier_alternative="mobilevit_xs",
            ),
            ModelSpec(
                name="mobilevit_xs",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=2.3,
                default_input_shape=[3, 256, 256],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 256,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.ANDROID],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="mobilevit_xs.cvnets_in1k",
                neighbor_models=["mobilevit_xxs", "mobilevit_s"],
                lighter_alternative="mobilevit_xxs",
                heavier_alternative="mobilevit_s",
            ),
            # EdgeNeXt
            ModelSpec(
                name="edgenext_xx_small",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=1.3,
                default_input_shape=[3, 256, 256],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 256,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="edgenext_xx_small.in1k",
                neighbor_models=["edgenext_x_small"],
                lighter_alternative=None,
                heavier_alternative="edgenext_x_small",
            ),
            ModelSpec(
                name="edgenext_x_small",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=2.3,
                default_input_shape=[3, 256, 256],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 256,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.ANDROID],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="edgenext_x_small.in1k",
                neighbor_models=["edgenext_xx_small", "edgenext_small"],
                lighter_alternative="edgenext_xx_small",
                heavier_alternative="edgenext_small",
            ),
            ModelSpec(
                name="convnext_tiny",
                family_id="timm",
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
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="convnext_tiny.fb_in22k_ft_in1k",
                neighbor_models=["resnet50", "efficientnet_b3"],
                lighter_alternative="resnet50",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="tf_efficientnetv2_b0",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=7.1,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="tf_efficientnetv2_b0.in1k",
                neighbor_models=["efficientnet_b1", "tf_efficientnetv2_b1"],
                lighter_alternative="efficientnet_b1",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="rexnet_100",
                family_id="timm",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=4.8,
                default_input_shape=[3, 224, 224],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 64,
                    "lr": 0.001,
                    "imgsz": 224,
                },
                pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.ANDROID, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [192, 224, 256],
                    "batch_size": [1],
                },
                source_url="https://github.com/huggingface/pytorch-image-models",
                pretrained_weights="rexnet_100.nav_in1k",
                neighbor_models=["mobilenetv3_large_100", "edgenext_x_small"],
                lighter_alternative="mobilenetv3_large_100",
                heavier_alternative="edgenext_x_small",
            ),
        ]
        specs.extend(self._build_extended_specs())
        for spec in specs:
            spec.edge_compatible = True
            if RuntimeId.TENSORRT not in spec.supported_runtimes:
                spec.supported_runtimes.append(RuntimeId.TENSORRT)
        return specs

    def _build_extended_specs(self) -> List[ModelSpec]:
        """Build an expanded timm catalog for edge-oriented profiling."""
        entries = [
            ("resnet101", 44.5, 224),
            ("resnet152", 60.2, 224),
            ("resnext50_32x4d", 25.0, 224),
            ("resnext101_32x8d", 88.8, 224),
            ("densenet121", 8.0, 224),
            ("densenet201", 20.0, 224),
            ("regnety_008", 6.4, 224),
            ("regnety_016", 11.2, 224),
            ("regnetx_008", 6.2, 224),
            ("regnetx_016", 9.2, 224),
            ("tf_efficientnet_b4", 19.3, 380),
            ("tf_efficientnet_b5", 30.0, 456),
            ("tf_efficientnetv2_s", 22.0, 384),
            ("mobilenetv4_conv_small.e2400_r224_in1k", 4.0, 224),
            ("mobilenetv4_conv_medium.e500_r256_in1k", 8.0, 256),
            ("convnext_small", 50.2, 224),
            ("convnext_base", 88.6, 224),
            ("vit_tiny_patch16_224", 5.7, 224),
            ("vit_small_patch16_224", 22.1, 224),
            ("maxxvit_tiny_tf_224", 31.0, 224),
        ]
        model_aliases = {
            # timm registry uses maxvit_* naming; keep model_id stable but map at runtime.
            "maxxvit_tiny_tf_224": "maxvit_tiny_tf_224.in1k",
        }
        specs: List[ModelSpec] = []
        for name, params_m, imgsz in entries:
            model_ref = model_aliases.get(name, name)
            specs.append(
                ModelSpec(
                    name=name,
                    family_id="timm",
                    modality=Modality.VISION,
                    supported_tasks=[TaskType.CLASSIFICATION],
                    params_m=params_m,
                    default_input_shape=[3, imgsz, imgsz],
                    default_hyperparams={"epochs": 50, "batch_size": 32, "lr": 0.001, "imgsz": imgsz},
                    pip_packages=["timm>=0.9.0", "torch", "torchvision"],
                    supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                    supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                    edge_compatible=True,
                    recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                    benchmark_input_signature={"imgsz": [imgsz], "batch_size": [1]},
                    source_url="https://huggingface.co/docs/timm/",
                    pretrained_weights=model_ref,
                    neighbor_models=[],
                    lighter_alternative=None,
                    heavier_alternative=None,
                )
            )
        return specs

    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py for timm models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}

        return f'''#!/usr/bin/env python3
"""Training script for {spec.name} classification model."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import yaml

# Configuration
model_name = "{spec.name}"
epochs = {hp.get("epochs", 50)}
batch_size = {hp.get("batch_size", 32)}
lr = {hp.get("lr", 0.001)}
imgsz = {hp.get("imgsz", 224)}
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
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
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

# Final output
output = {{
    "status": "success",
    "model_path": "outputs/best.pt",
    "metrics": {{
        "accuracy": best_acc,
        "accuracy_top1": best_acc,
    }},
}}
print(json.dumps(output))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py for timm models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        imgsz = hp.get("imgsz", 224)

        return f'''#!/usr/bin/env python3
"""Inference benchmark script for {spec.name} classification model."""
import json
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch
import timm
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = Path(os.environ.get("EDGECRAFT_ARTIFACT_PATH", "outputs/best.pt"))
if not model_path.exists() and Path("outputs/best.pt").exists():
    model_path = Path("outputs/best.pt")
if not model_path.exists():
    print(json.dumps({{"status": "error", "error": "Model file not found", "artifact_used": str(model_path)}}))
    sys.exit(1)

# Configuration
model_name = "{spec.name}"
imgsz = {imgsz}
cfg_path = Path("config/data.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {{}}
cfg = cfg or {{}}
num_classes = int(cfg.get("num_classes") or cfg.get("nc") or 10)

# Create model and load weights
checkpoint = torch.load(str(model_path), map_location=device)
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    num_classes = int(checkpoint.get("num_classes") or num_classes)
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# Export to ONNX if requested
export_format = "{context.export_format.value}"
if export_format == "onnx":
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    torch.onnx.export(
        model, dummy, "outputs/best.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch"}}}},
    )

# Warmup
dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
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

# Memory estimation
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
        if runtime_id == RuntimeId.ONNXRUNTIME:
            _script = f'''#!/usr/bin/env python3
import json
import time
import numpy as np

try:
    import timm
    import torch
    import onnxruntime as ort
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

model_name = "{model_spec.name}"
model_ref = "{model_spec.pretrained_weights or model_spec.name}"
imgsz = {imgsz}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    candidates = []
    for c in (model_ref, model_name, model_name.replace("maxxvit_", "maxvit_")):
        if c and c not in candidates:
            candidates.append(c)
        if c and "." in c:
            base = c.split(".", 1)[0]
            if base not in candidates:
                candidates.append(base)
    model = None
    last_error = None
    for cand in candidates:
        try:
            model = timm.create_model(cand, pretrained=False)
            break
        except Exception as e:
            last_error = e
    if model is None:
        raise RuntimeError(f"Unknown model ({{model_name}}); tried: {{candidates}}; last_error={{last_error}}")
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch"}}}},
        opset_version=17,
    )
    sess = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

dummy_np = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
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
            return wrap_with_tegrastats(_script)
        if runtime_id == RuntimeId.TENSORRT:
            return wrap_with_tegrastats(self._render_profile_tensorrt(model_spec, imgsz, warmup, iterations))
        return ""

    def _render_profile_tensorrt(
        self,
        model_spec: ModelSpec,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import re
import shutil
import subprocess

try:
    import timm
    import torch
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

trtexec = shutil.which("trtexec")
if not trtexec:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":"trtexec not found in PATH"}}))
    raise SystemExit(1)

model_name = "{model_spec.name}"
model_ref = "{model_spec.pretrained_weights or model_spec.name}"
imgsz = {imgsz}
try:
    candidates = []
    for c in (model_ref, model_name, model_name.replace("maxxvit_", "maxvit_")):
        if c and c not in candidates:
            candidates.append(c)
        if c and "." in c:
            base = c.split(".", 1)[0]
            if base not in candidates:
                candidates.append(base)
    model = None
    last_error = None
    for cand in candidates:
        try:
            model = timm.create_model(cand, pretrained=False).eval().cpu()
            break
        except Exception as e:
            last_error = e
    if model is None:
        raise RuntimeError(f"Unknown model ({{model_name}}); tried: {{candidates}}; last_error={{last_error}}")
    dummy = torch.randn(1, 3, imgsz, imgsz)
    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch"}}}},
        opset_version=17,
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export","error_message":str(e)}}))
    raise SystemExit(1)

cmd = [
    trtexec,
    "--onnx=model.onnx",
    "--saveEngine=model.engine",
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


# Module-level instance for easy access
timm_plugin = TimmPlugin()
