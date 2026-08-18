# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ros:humble

SHELL ["/bin/bash", "-c"]

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       nano \
                       vim \
                       python3-pip \
                       libeigen3-dev \
                       tmux \
                       ros-humble-rviz2
RUN apt-get -y dist-upgrade
RUN pip3 install transforms3d

# f1tenth gym
# Upstream f1tenth_gym pins legacy gym (==0.19.0) and numpy<=1.22, whose
# packaging metadata fails to install under the newer pip/setuptools that a
# Python 3.10 (Humble) base can carry.  Pin the build frontend to versions
# that still resolve the legacy editable install (the well-known gym
# 0.19/0.21 packaging bug); these also build the ament_python package below.
RUN pip3 install "pip<24.1" "setuptools==65.5.0" "wheel<0.40.0"
RUN git clone https://github.com/f1tenth/f1tenth_gym
RUN cd f1tenth_gym && \
    pip3 install -e .

# ros2 gym bridge
RUN mkdir -p sim_ws/src/f1tenth_gym_ros
COPY . /sim_ws/src/f1tenth_gym_ros
RUN source /opt/ros/humble/setup.bash && \
    cd sim_ws/ && \
    apt-get update --fix-missing && \
    rosdep install -i --from-path src --rosdistro humble -y && \
    colcon build

# torch, for the learned policies (tools/train_rl.py, tools/train_duel.py).
# It belongs in the image rather than being pip-installed into a running
# container, or every rebuild silently loses the ability to train.
#
# This MUST come after colcon build.  torch pulls in a modern setuptools,
# which calls packaging.canonicalize_version(strip_trailing_zero=...) -- a
# keyword the older `packaging` in this image does not accept -- and that
# combination makes colcon build fail with:
#     TypeError: canonicalize_version() got an unexpected keyword argument
# Nothing is built after this point, so the newer setuptools is harmless,
# and `packaging` is upgraded alongside it so a LATER in-container
# `colcon build` still works (the repo is bind-mounted and does get rebuilt).
#
# CPU build deliberately: the deployed actor is ~113k parameters and runs in
# well under a millisecond on CPU, and the training bottleneck is the gym
# simulation, which is CPU-bound regardless -- a CUDA image would multiply
# the size for no useful speedup.
RUN pip3 install --upgrade pip packaging && \
    pip3 install --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/bash"]
