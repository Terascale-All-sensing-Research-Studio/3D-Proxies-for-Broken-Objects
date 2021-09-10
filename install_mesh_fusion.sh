# With a newer gpu you may run into this issue:
# https://github.com/davidstutz/mesh-fusion/issues/6
#

# Install packages
echo "Installing packages ..."
sudo apt-get install -y cmake python3.6 python3.6-dev python3-pip freeglut3-dev libgl1-mesa-dev libglew-dev zlib1g-dev libgtest-dev libeigen3-dev build-essential libgles2-mesa-dev libegl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols

# Build the libraries
mkdir libs && \
    cd libs && \
    git clone https://github.com/davidstutz/mesh-fusion.git && \
    cd mesh-fusion && \
    cd libfusioncpu && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd .. && \
    python setup.py build_ext --inplace && \
    pip install . && \
    cd .. && \
    cd librender && \
    python setup.py build_ext --inplace && \
    mv pyrender.cpython-36m-x86_64-linux-gnu.so pyrender.so && \
    pip install . && \
    cd .. && \
    cd libmcubes && \
    python setup.py build_ext --inplace