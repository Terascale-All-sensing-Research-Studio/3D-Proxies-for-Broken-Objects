# 3D-Proxies-for-Broken-Objects

Code for "Using Learned Visual and Geometric Features to Retrieve Complete 3D Proxies for Broken Objects." Published at ACM SCF 2021. 

## Installation

The code can be split into three sequential stages that must be performed before the database can be queried:

1) Normalization and Breaking,
2) Feature Extraction,
3) Database Creation.

Unfortunately some of these steps require more than one environment. The environments are listed below:

- Docker environment: Handles object breaking (requires pymesh).
- Anaconda environment: Hanldes database creation and querying (requires faiss).
- Pip-torch environment: Handles pointnet++ feature extraction (requires torch).
- Pip-tensorflow environment: Handles pretty much everything else.

Instructions to setup and use each of the environments can be found below. Tested on Ubuntu 18.04 with python3.6

Note: the environment activation scripts reference the environment variable `"$DATADIR"`, and will expect it to be defined in a gitignored file `constants.sh`. The path to pointnet++ and mesh-fusion dependancies should also be defined here. Here's an example of what it should look like:

```
export DATADIR="/media/DATACENTER"                      # Where the data lives
export POINTNETPATH=`pwd`"/Pointnet_Pointnet2_pytorch/" # Where the pointnet++ implementation lives
export PYTHONPATH=$PYTHONPATH:`pwd`/libs/mesh-fusion    # Where the meshfusion implementation lives
```

### Docker Environment

Once you have docker installed, just run the docker activation script.

```
# Lauch the environment with an interactive terminal
./activate_docker.bash

# cd back to the working directory (you should be back in the repo)
cd /opt/ws/python
```

### Anaconda environment

Make sure you have anaconda or miniconda installed and then run the following:

```
# Create the environment
conda create --name 3dp --file requirements_conda.txt

# Now install faiss
conda install -c pytorch faiss-gpu
```

### Pip-torch environment

Make sure you have virtualenv installed and then run the following:

```
# Create the environment
virtualenv -p python3 env-torch
source activate_torch.sh

# Install requirements
pip install -r requirements_torch.txt
```

Running pointnet++ also requires the pointnet implementation provided by [Xu Yan](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Clone that repository and then add the path to the `constants.sh` file (see above). 

```
# Clone the repo 
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
```

### Pip-tensorflow environment

Make sure you have virtualenv installed and then run the following:

```
# Create the environment
virtualenv -p python3 env-tf
source activate_tf.sh

# Install requirements
pip install -r requirements_tf.txt
```

Running waterproofing also requires the librender and libfusion libraries provided by [David Stutz](https://github.com/davidstutz/mesh-fusion). We've provided a script to get this up and running automatically but it may not work on all systems so pay close attention to the output and diagnose any errors. It will create a new `libs` directory, which should also be added to your python path using the `constants.sh` file (see above).

```
# Run the install script
./install_mesh_fusion.sh
```

## Usage

All scripts are stored in the python directory.

### Preprocessing

The main.py script handles normalization, breaking, and feature extraction. It must be passed the database location, the splits file that you'll be using, and the operations that you'd like to perform e.g.:

```
python main.py ShapeNetCore.v2 ShapeNetCore.v2/splits.json WATERPROOF
```

The operations that may be passed are:
- WATERPROOF: Run waterproofing with textures.
- CLEAN: Run laplacian smoothing. 
- BREAK: Break mesh. (Requires the docker environment.)
- RENDER: Render mesh from multiple viewpoints. 
- DEPTH: Render depth image of mesh from multiple viewpoints.
- DEPTH_PANO: Render depth panoramic images from multiple viewpoints.
- FCGF: Get Fully Convolutional Geometric Features.
- SIFT: Get SIFT features. 
- SIFT_PANO: Get SIFT from panoramas.
- ORB: Get ORB features. 
- global_VGG: Get VGG16 features.
- global_POINTNET: Get PointNet++ features. (Requires the pip-torch environment.)

Run it with the help flag (`main.py --help`) for additional information on arguments.

### Databse Creation and Querying

Step one is to build the database:

```
create_database.py <path-to-ShapeNetCore.v2> <path-to-splits-file> <where-to-save-database> <list-of-features>
```

With the database created, you should be able to access attributes like this:
```python
from inshapenet.database import ObjectDatabase
odb = ObjectDatabase(load_from="/path/to/save/location")

# Get a ShapenetObject corresponding to the first query object
idx = 0
obj = odb.get_object_query(idx)

# Load the first complete object, upsampled
complete_pts = np.load(obj.path_model_c_upsampled())['features']

# Load the first broken object, upsampled
complete_pts = np.load(obj.path_model_b_upsampled())['features']

# Load the features corresponding to the first query object
feats = odb.get_feature_query_c(idx)

# Query those feats into the database
result = odb.hierarchical_query(feats)
print('Score summary:')
for r in result:
    print(r[0], r[1]['score'])
```