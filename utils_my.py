import os
import os.path as op
# sym-link
if not op.exists('data/MFQEv2'):
    if not op.exists('data/'):
        os.system("mkdir data/")
    os.system(f"ln -s {'/data/lk/datasets/MFQEv2_dataset'} ./data/MFQEv2")
    print("Sym-linking done.")
else:
    print("data/MFQEv2 already exists.")
