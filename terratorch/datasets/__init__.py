# Copyright contributors to the Terratorch project

from terratorch.datasets.biomassters import BioMasstersNonGeo
from terratorch.datasets.burn_intensity import BurnIntensityNonGeo
from terratorch.datasets.carbonflux import CarbonFluxNonGeo
from terratorch.datasets.fire_scars import FireScarsHLS, FireScarsNonGeo, FireScarsSegmentationMask
from terratorch.datasets.forestnet import ForestNetNonGeo
from terratorch.datasets.generic_multimodal_dataset import (
    GenericMultimodalDataset,
    GenericMultimodalPixelwiseRegressionDataset,
    GenericMultimodalScalarDataset,
    GenericMultimodalSegmentationDataset,
)
from terratorch.datasets.generic_nongeo_od_dataset import (
    GenericNonGeoObjectDetectionDataset,
)
from terratorch.datasets.generic_pixel_wise_dataset import (
    GenericNonGeoPixelwiseRegressionDataset,
    GenericNonGeoSegmentationDataset,
)
from terratorch.datasets.generic_scalar_label_dataset import (
    GenericNonGeoClassificationDataset,
)
from terratorch.datasets.hls import HLSL30, HLSS30
from terratorch.datasets.landslide4sense import Landslide4SenseNonGeo
from terratorch.datasets.m_bigearthnet import MBigEarthNonGeo
from terratorch.datasets.m_brick_kiln import MBrickKilnNonGeo

# geobench datasets segmentation
from terratorch.datasets.m_cashew_plantation import MBeninSmallHolderCashewsNonGeo
from terratorch.datasets.m_chesapeake_landcover import MChesapeakeLandcoverNonGeo

# geobench datasets classification
from terratorch.datasets.m_eurosat import MEuroSATNonGeo
from terratorch.datasets.m_forestnet import MForestNetNonGeo
from terratorch.datasets.m_neontree import MNeonTreeNonGeo
from terratorch.datasets.m_nz_cattle import MNzCattleNonGeo
from terratorch.datasets.m_pv4ger import MPv4gerNonGeo
from terratorch.datasets.m_pv4ger_seg import MPv4gerSegNonGeo
from terratorch.datasets.m_SA_crop_type import MSACropTypeNonGeo
from terratorch.datasets.m_so2sat import MSo2SatNonGeo
from terratorch.datasets.m_VHR10 import mVHR10
from terratorch.datasets.multi_temporal_crop_classification import MultiTemporalCropClassification
from terratorch.datasets.open_sentinel_map import OpenSentinelMap

# miscellaneous datasets
from terratorch.datasets.openearthmap import OpenEarthMapNonGeo
from terratorch.datasets.pastis import PASTIS

# GenericNonGeoRegressionDataset,
from terratorch.datasets.sen1floods11 import Sen1Floods11NonGeo

# from terratorch.datasets.sen1floods11 import Sen1Floods11NonGeo
from terratorch.datasets.sen4agrinet import Sen4AgriNet
from terratorch.datasets.tiled_dataset_wrapper import TiledDataset

# Generic Classification Dataset
from terratorch.datasets.sen4map import Sen4MapDatasetMonthlyComposites
from terratorch.datasets.substation import Substation
from terratorch.datasets.utils import HLSBands, OpticalBands, SARBands, unbind_samples

# TorchGeo RasterDatasets
from terratorch.datasets.embedding_dataset import EmbeddingDataset
from terratorch.datasets.wsf import WSF2019, WSFEvolution

# Surya datasets
try:
    from terratorch.datasets.helio import HelioNetCDFDataset
except ImportError:
    pass

__all__ = (
    "GenericNonGeoSegmentationDataset",
    "GenericNonGeoPixelwiseRegressionDataset",
    "GenericNonGeoClassificationDataset",
    "GenericNonGeoObjectDetectionDataset",
    # "GenericNonGeoRegressionDataset",
    "GenericMultimodalDataset",
    "GenericMultimodalSegmentationDataset",
    "GenericMultimodalPixelwiseRegressionDataset",
    "GenericMultimodalScalarDataset",
    "GenericNonGeoRegressionDataset",
    "BurnIntensityNonGeo",
    "CarbonFluxNonGeo",
    "Landslide4SenseNonGeo",
    "BioMasstersNonGeo",
    "ForestNetNonGeo",
    "FireScarsNonGeo",
    "FireScarsHLS",
    "FireScarsSegmentationMask",
    "Sen1Floods11NonGeo",
    "MultiTemporalCropClassification",
    "Sen4MapDatasetMonthlyComposites",
    "HLSBands",
    "MEuroSATNonGeo",
    "MBigEarthNonGeo",
    "MBrickKilnNonGeo",
    "MForestNetNonGeo",
    "MSo2SatNonGeo",
    "MPv4gerNonGeo",
    "MBeninSmallHolderCashewsNonGeo",
    "MNzCattleNonGeo",
    "MChesapeakeLandcoverNonGeo",
    "MPv4gerSegNonGeo",
    "MSACropTypeNonGeo",
    "MNeonTreeNonGeo",
    "OpenSentinelMap",
    "PASTIS",
    "Sen4AgriNet",
    "TiledDataset",
    "WSF2019",
    "WSFEvolution",
    "HLSL30",
    "HLSS30",
    "OpticalBands",
    "SARBands",
    "OpenEarthMapNonGeo",
    "mVHR10",
    "Substation",
    "HelioNetCDFDataset",
    "EmbeddingDataset",
    "unbind_samples",
)
