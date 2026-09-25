
Model factories build the model that is fine-tuned by TerraTorch. Specifically, a backbone is used an encoder and combined with a task-specific decoder and head. 
Necks are using to reshape the encoder output to be compatible with the decoder input.

!!! tip
    The `EncoderDecoderFactory` is the default factory for segmentation, pixel-wise regression, and classification tasks.

Other commonly used factories are the `ObjectDetectionModelFactory` for object detection tasks and sometimes the `FullModelFactory` if a model is registered in the `FULL_MODEL_REGISTRY` and can be directly applied to a specific task. 

:::terratorch.models.encoder_decoder_factory.EncoderDecoderFactory

:::terratorch.models.object_detection_model_factory.ObjectDetectionModelFactory

:::terratorch.models.full_model_factory.FullModelFactory

:::terratorch.models.smp_model_factory.SMPModelFactory

:::terratorch.models.timm_model_factory.TimmModelFactory

:::terratorch.models.generic_model_factory.GenericModelFactory

:::terratorch.models.clay_model_factory.ClayModelFactory

:::terratorch.models.generic_unet_model_factory.GenericUnetModelFactory


Base factory: 

::: terratorch.models.model.ModelFactory

[//]: # (:::terratorch.models.prithvi_model_factory.PrithviModelFactory)
