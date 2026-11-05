# Copyright contributors to the Terratorch project

import logging

from terratorch.datamodules.era5 import ERA5DataModule
from terratorch.datamodules.fire_scars import FireScarsNonGeoDataModule
from terratorch.datamodules.generic_nongeo_od_data_module import (
    GenericNonGeoObjectDetectionDataModule,
)
from terratorch.datamodules.generic_pixel_wise_data_module import (
    GenericNonGeoPixelwiseRegressionDataModule,
    GenericNonGeoSegmentationDataModule,
)
from terratorch.datamodules.embedding_classification_data_module import (
    EmbeddingClassificationDataModule,
)
from terratorch.datamodules.generic_scalar_label_data_module import (
    GenericNonGeoClassificationDataModule,
)
from terratorch.datamodules.m_bigearthnet import MBigEarthNonGeoDataModule
from terratorch.datamodules.m_brick_kiln import MBrickKilnNonGeoDataModule

# geobench segmentation datamodules
from terratorch.datamodules.m_cashew_plantation import MBeninSmallHolderCashewsNonGeoDataModule
from terratorch.datamodules.m_chesapeake_landcover import MChesapeakeLandcoverNonGeoDataModule

# geobench classification datamodules
from terratorch.datamodules.m_eurosat import MEuroSATNonGeoDataModule
from terratorch.datamodules.m_forestnet import MForestNetNonGeoDataModule
from terratorch.datamodules.m_neontree import MNeonTreeNonGeoDataModule
from terratorch.datamodules.m_nz_cattle import MNzCattleNonGeoDataModule
from terratorch.datamodules.m_pv4ger import MPv4gerNonGeoDataModule
from terratorch.datamodules.m_pv4ger_seg import MPv4gerSegNonGeoDataModule
from terratorch.datamodules.m_SA_crop_type import MSACropTypeNonGeoDataModule
from terratorch.datamodules.m_so2sat import MSo2SatNonGeoDataModule
from terratorch.datamodules.multi_temporal_crop_classification import MultiTemporalCropClassificationDataModule
from terratorch.datamodules.open_sentinel_map import OpenSentinelMapDataModule
from terratorch.datamodules.pastis import PASTISDataModule

try:
    wxc_present = True
    from terratorch.datamodules.merra2_downscale import Merra2DownscaleNonGeoDataModule
except ImportError:
    logging.getLogger("terratorch").debug("wxc_downscaling not installed")
    wxc_present = False

# GenericNonGeoRegressionDataModule,
from terratorch.datamodules.biomassters import BioMasstersNonGeoDataModule
from terratorch.datamodules.burn_intensity import BurnIntensityNonGeoDataModule
from terratorch.datamodules.carbonflux import CarbonFluxNonGeoDataModule
from terratorch.datamodules.forestnet import ForestNetNonGeoDataModule
from terratorch.datamodules.generic_multimodal_data_module import GenericMultiModalDataModule
from terratorch.datamodules.landslide4sense import Landslide4SenseNonGeoDataModule

# miscellaneous datamodules
from terratorch.datamodules.openearthmap import OpenEarthMapNonGeoDataModule
from terratorch.datamodules.sen1floods11 import Sen1Floods11NonGeoDataModule
from terratorch.datamodules.sen4agrinet import Sen4AgriNetDataModule
from terratorch.datamodules.tiled_datamodule_wrapper import TilingDataModuleWrapper
from terratorch.datamodules.torchgeo_data_module import TorchGeoDataModule, TorchNonGeoDataModule

try:
    from terratorch.datamodules.geobench_v2_data_module import (
        GeoBenchV2ClassificationDataModule,
        GeoBenchV2ObjectDetectionDataModule,
        GeoBenchV2SegmentationDataModule,
    )

    geobench_v2_present = True
except ImportError:
    logging.getLogger("terratorch").debug("geobench_v2 not installed")
    geobench_v2_present = False

# miscellaneous datamodules

# Generic classification datamodule
from terratorch.datamodules.m_VHR10 import mVHR10DataModule
from terratorch.datamodules.sen4map import Sen4MapLucasDataModule
from terratorch.datamodules.substation import SubstationDataModule

try:
    from terratorch.datamodules.helio import HelioNetCDFDataModule
except ImportError:
    pass

__all__ = [
    "EmbeddingClassificationDataModule",
    "GenericNonGeoSegmentationDataModule",
    "GenericNonGeoPixelwiseRegressionDataModule",
    "GenericNonGeoClassificationDataModule",
    "GenericNonGeoObjectDetectionDataModule",
    # "GenericNonGeoRegressionDataModule",
    "BurnIntensityNonGeoDataModule",
    "CarbonFluxNonGeoDataModule",
    "Landslide4SenseNonGeoDataModule",
    "ForestNetNonGeoDataModule",
    "BioMasstersNonGeoDataModule",
    "Sen1Floods11NonGeoDataModule",
    "Sen4MapLucasDataModule",
    "FireScarsNonGeoDataModule",
    "MultiTemporalCropClassificationDataModule",
    "TorchGeoDataModule",
    "TorchNonGeoDataModule",
    "TilingDataModuleWrapper",
    "MEuroSATNonGeoDataModule",
    "MBigEarthNonGeoDataModule",
    "MBrickKilnNonGeoDataModule",
    "MForestNetNonGeoDataModule",
    "MSo2SatNonGeoDataModule",
    "MPv4gerNonGeoDataModule",
    "MBeninSmallHolderCashewsNonGeoDataModule",
    "MNzCattleNonGeoDataModule",
    "MChesapeakeLandcoverNonGeoDataModule",
    "MPv4gerSegNonGeoDataModule",
    "MSACropTypeNonGeoDataModule",
    "MNeonTreeNonGeoDataModule",
    "OpenEarthMapNonGeoDataModule",
    "OpenSentinelMapDataModule",
    "PASTISDataModule",
    "Sen4AgriNetDataModule",
    "GenericMultiModalDataModule",
    "mVHR10DataModule",
    "SubstationDataModule",
    "HelioNetCDFDataModule",
]

if wxc_present:
    __all__.append("Merra2DownscaleNonGeoDataModule")


if geobench_v2_present:
    __all__.extend(
        [
            "GeoBenchV2ClassificationDataModule",
            "GeoBenchV2ObjectDetectionDataModule",
            "GeoBenchV2SegmentationDataModule",
        ]
    )
