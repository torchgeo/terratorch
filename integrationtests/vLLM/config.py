models = {
    "prithvi_300m_sen1floods11": {
        "location": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "io_processor_plugin": "terratorch_segmentation",
    },
    "prithvi_300m_burnscars": {
        "location": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        "io_processor_plugin": "terratorch_segmentation",
    },
    "terramind_base_flood": {
        "location": "ibm-esa-geospatial/TerraMind-base-Flood",
        "io_processor_plugin": "terratorch_tm_segmentation",
    },
    "terramind_base_fire": {
        "location": "ibm-esa-geospatial/TerraMind-base-Fire",
        "io_processor_plugin": "terratorch_tm_segmentation",
    },
}

# Expected output hash for each model/input image (independent of input/output format)
models_output = {
    "prithvi_300m_sen1floods11": {
        "india": "f7dc282de2c36942",
        "valencia": "aa6d92ad25926a5e",
    },
    "prithvi_300m_burnscars": {
        "burnscars": "c17c4f602ea7b616",
    },
    "terramind_base_flood": {
        "flood": "dc25f59d25cc0273",
    },
    "terramind_base_fire": {
        "fire": "f6088cf60b1de4ea",
    },
}

# Map of input image names to their configurations
input_images = {
    "india": {
        "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
        "indices": [1, 2, 3, 8, 11, 12],
    },
    "valencia": {
        "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
    },
    "burnscars": {
        "image_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/resolve/main/examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif",
    },
    "flood": {
        "image_url": {
            "DEM": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/flood/EMSR354_23_37LFH_DEM.tif",
            "S1RTC": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/flood/EMSR354_23_37LFH_S1RTC.zarr.zip",
            "S2L2A": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/flood/EMSR354_23_37LFH_S2L2A.zarr.zip",
        },
    },
    "fire": {
        "image_url": {
            "DEM": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/fire/EMSR686_1_35TMF_x413905_y4537585_DEM.tif",
            "S1RTC": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/fire/EMSR686_1_35TMF_x413905_y4537585_S1RTC.zarr.zip",
            "S2L2A": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/fire/EMSR686_1_35TMF_x413905_y4537585_S2L2A.zarr.zip",
        },
    },
}
