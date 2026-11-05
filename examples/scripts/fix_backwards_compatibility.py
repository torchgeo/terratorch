"""
This example is deprecated since TerraTorch >= v1.2.
We're porting this example to a new set of nice and clean examples for you.

In the meanwhile please checkout the config files under:
./tests/resources/configs
"""

print(__doc__)
#!/usr/bin/env python3
import argparse
import os
import torch

"""
usage:
python3 ./scripts/fix_backwards_copatibility.py  <file>
"""

homedir = os.path.expanduser('~')
cwd = os.getcwd()

# Get filename of checkpoint or model file to correct
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert model for backwards compatibility on terratorch versions 0.99 or higher')
    parser.add_argument('file',
                        action='store',
                        metavar='INPUT_FILE',
                        type=str,
                        help='Checkpoint file or model to be corrected for backwards compatibility on terratorch versions 0.99 or higher')
    arg = parser.parse_args()

# Input file
path_in = arg.file
print('path in:', path_in)
path_out = (arg.file).split('.')[0]+'_Fixed.'+(arg.file).split('.')[1]
print('path out:', path_out)

state_dict = torch.load(path_in, map_location=torch.device('cpu'), weights_only=True)
state_dict_renamed = {}

for k, v in state_dict.items():
    # remove the module. part
    if k == 'state_dict':
        state_dict_renamed[k] = {}
        for k1, v1 in v.items():
            splits = k1.split(".")
            splits_ = [s for s in splits if "timm" not in s]
            k1_ = ".".join(splits_)
            if k1 != k1_:
                state_dict_renamed[k][k1_] = v1
            else:
                state_dict_renamed[k][k1] = v1
    else:
        state_dict_renamed[k] = v

torch.save(state_dict_renamed, path_out)
