
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/aditya/pushmeet_bias/mosek/6/tools/platform/linux64x86/bin
export MOSEKLM_LICENSE_FILE=~/aditya/pushmeet_bias/mosek/6/licenses/mosek.lic
alias cv="python /afs/csail.mit.edu/u/b/bzhou/scripts/check_cluster.py vision"
alias synthetic="cd /data/vision/torralba/3d-object-recog/project/synthetic_training/aditya_code"
#export PATH=/data/vision/torralba/aditya_datasets/matlab/bin:$HOME/python/bin:$PATH
#export PYTHONUSERBASE=$HOME/python

#export LD_LIBRARY_PATH=/data/vision/torralba/commonsense/cudnnv4/cuda/lib64:$LD_LIBRARY_PATH

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfreetype.so.6 matlab
#export LD_LIBRARY_PATH=/data/vision/oliva/scenedataset/lib/cuda-8.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vision/torralba/datasetbias/lib/glog/0.3.3/lib
export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/boost/1.57.0/stage/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/boost/1.54.0/stage/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/cuda/6.5/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/aditya_datasets/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/oliva/scenedataset/segmentation_hang/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
#export MKL_RT_LIBRARY=/data/vision/oliva/scenedataset/segmentation_hang/intel/mkl/lib/intel64
#export MKL_INCLUDE_DIR=/data/vision/oliva/scenedataset/segmentation_hang/intel/mkl/include
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/protobuf/2.4.1-matlab/src/.libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/mdb/libraries/liblmdb/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/cudnn/6.5-v2:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/opencv/2.4.5/install/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/oliva/scenedataset/lib/cuda-8.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/scratch/billf/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/cuda/7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vision/torralba/datasetbias/lib/protobuf/2.5.0/src/.libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vision/torralba/datasetbias/lib/protobuf/2.4.1/src/.libs
export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/lib/ubuntu_12.04_lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/lib/x86_64-linux-gnu/:/data/vision/torralba/aditya_datasets/anaconda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/projectx/libdispatch/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/regionmem/memorability_cnn/lib/matlab-json/json-c/.libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/datasetbias/projectx/lib/DeepBeliefSDK/source:$LD_LIBRARY_PATH
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/lib/x86_64-linux-gnu/libgcc_s.so.1:$LD_PRELOAD
export PATH=/data/vision/torralba/aditya_datasets/matlab/bin:$PATH
export KMP_DUPLICATE_LIB_OK=TRUE
export HDF5_DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=8
#export PATH=$PATH:/data/vision/oliva/scenedataset/lib/cuda-8.0/bin:/usr/bin
export PATH=$PATH:/data/vision/scratch/billf/local/cuda-8.0/bin:/usr/bin
export PATH=/data/vision/torralba/datasetbias/lib/protobuf/2.5.0/src/.libs:$PATH
alias "c=xclip -selection clipboard"
alias "v=xclip -o"
alias dataaug="cd /data/vision/torralba/aditya_datasets/data_aug"
alias co="cd /data/vision/torralba/datasetbias/comsol/comsol44/bin"
alias matlab-latest="/data/vision/torralba/datasetbias/matlab-2014a/bin/matlab"
#alias matlab="/data/vision/torralba/datasetbias/matlab-2014a/bin/matlab"


#export CUDA_HOME=/data/vision/oliva/scenedataset/lib/cuda-8.0
export CUDA_HOME=/data/vision/scratch/billf/local/cuda-8.0
export CAFFE_HOME=/data/vision/torralba/segmentation/places/PSPNet

#export LD_LIBRARY_PATH=/data/vision/oliva/scenedataset/lib/cuda/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/vision/oliva/scenedataset/lib/cuda/lib:$LD_LIBRARY_PATH
export PATH=/data/vision/oliva/scenedataset/anaconda2/bin:$PATH

. /data/vision/torralba/deepscene/distro/install/bin/torch-activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vision/billf/object-properties/local/lib
export PYTHONPATH=$PYTHONPATH:/data/vision/torralba/segmentation/places/PSPNet/python

source /data/vision/torralba/aditya_datasets/intel/mkl/bin/mklvars.sh intel64
export LD_LIBRARY_PATH=/data/vision/oliva/scenedataset/lib/cudnn-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/vision/torralba/segmentation/places/PSPNet/build/lib:$LD_LIBRARY_PATH
