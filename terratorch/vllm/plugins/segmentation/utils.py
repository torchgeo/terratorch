import base64
import urllib.request
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
import aiohttp
from anyio import open_file
from terratorch.cli_tools import is_one_band
from rasterio import MemoryFile

async def download_file_async(url: str, dest_file: Path | None = None) -> BytesIO | None:
    async with (aiohttp).ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise an error for bad responses
            data = await response.read()
            if dest_file:
                dest_file.write_bytes(data)
            else:
                return BytesIO(data)
                
async def read_file_async(path: str) -> BytesIO:
    async with await open_file(path, "rb") as f:
        contents = await f.read()
        return BytesIO(contents)

def download_file_sync(url: str, dest_file: Path | None = None) -> BytesIO | None:
    with urllib.request.urlopen(url) as response:
        data = response.read()
    if dest_file:
        dest_file.write_bytes(data)
    else:
        return BytesIO(data)


def get_filename_from_url(url: str) -> str:
    """
    Extracts the file name from the path of a given URL.
    Returns an empty string if no file name is present.
    """
    parsed_url = urlparse(url)
    return Path(parsed_url.path).name

@contextmanager
def path_or_tmpdir(prompt_dict):
    if prompt_dict.data_format == "path":
        yield prompt_dict.data
        return
    
    with TemporaryDirectory() as tmpdir:
        yield tmpdir

def to_base64_tiff(img_wrt: "torch.Tensor", metadata: dict) -> str:
    # Adapting the number of bands to be compatible with the
    # output dimensions.
    if not is_one_band(img_wrt):
        count = img_wrt.shape[0]
        metadata["count"] = count

    with MemoryFile() as mem_file:
        with mem_file.open(**metadata) as dest:
            if is_one_band(img_wrt):
                img_wrt = img_wrt[None]

            for i in range(img_wrt.shape[0]):
                dest.write(img_wrt[i, :, :], i + 1)
            
        tiff_bytes = mem_file.read()
    

    return base64.b64encode(tiff_bytes).decode('utf-8')