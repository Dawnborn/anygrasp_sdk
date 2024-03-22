# My Installation

conda create -n anygrasp python=3.9

source /data/hdd1/storage/junpeng/ws_anygrasp/prep_cuda.sh # cuda 10.2

pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 # pytorch 版本不行

------
```
conda create -n anygrasp2 python=3.9

source /data/hdd1/storage/junpeng/ws_anygrasp/prep_cuda.sh # cuda 11.1

conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

conda install openblas-devel -c anaconda

conda install numpy

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

```
cd pointnet2
python setup.py install
```
报错
```
No CUDA runtime is found, using CUDA_HOME='/storage/software/cuda/cuda-11.1'
/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/dist.py:265: UserWarning: Unknown distribution option: 'pacakges'
  warnings.warn(msg)
running install
/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/cmd.py:66: SetuptoolsDeprecationWarning: setup.py install is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` directly.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html for details.
        ********************************************************************************

!!
  self.initialize_options()
/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/cmd.py:66: EasyInstallDeprecationWarning: easy_install command is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` and ``easy_install``.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://github.com/pypa/setuptools/issues/917 for details.
        ********************************************************************************

!!
  self.initialize_options()
running bdist_egg
running egg_info
writing pointnet2.egg-info/PKG-INFO
writing dependency_links to pointnet2.egg-info/dependency_links.txt
writing top-level names to pointnet2.egg-info/top_level.txt
reading manifest file 'pointnet2.egg-info/SOURCES.txt'
writing manifest file 'pointnet2.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_py
copying pointnet2/pointnet2_utils.py -> build/lib.linux-x86_64-cpython-39/pointnet2
copying pointnet2/__init__.py -> build/lib.linux-x86_64-cpython-39/pointnet2
copying pointnet2/pointnet2_modules.py -> build/lib.linux-x86_64-cpython-39/pointnet2
copying pointnet2/pytorch_utils.py -> build/lib.linux-x86_64-cpython-39/pointnet2
running build_ext
building 'pointnet2._ext' extension
Traceback (most recent call last):
  File "/storage/user/lhao/hjp/ws_anygrasp/anygrasp_sdk/pointnet2/setup.py", line 18, in <module>
    setup(
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/__init__.py", line 104, in setup
    return distutils.core.setup(**attrs)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/core.py", line 185, in setup
    return run_commands(dist)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/core.py", line 201, in run_commands
    dist.run_commands()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 969, in run_commands
    self.run_command(cmd)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/command/install.py", line 87, in run
    self.do_egg_install()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/command/install.py", line 139, in do_egg_install
    self.run_command('bdist_egg')
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
    self.distribution.run_command(command)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/command/bdist_egg.py", line 167, in run
    cmd = self.call_command('install_lib', warn_dir=0)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/command/bdist_egg.py", line 153, in call_command
    self.run_command(cmdname)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
    self.distribution.run_command(command)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/command/install_lib.py", line 11, in run
    self.build()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/command/install_lib.py", line 111, in build
    self.run_command('build_ext')
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
    self.distribution.run_command(command)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/command/build_ext.py", line 345, in run
    self.build_extensions()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 709, in build_extensions
    build_ext.build_extensions(self)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/command/build_ext.py", line 467, in build_extensions
    self._build_extensions_serial()
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/command/build_ext.py", line 493, in _build_extensions_serial
    self.build_extension(ext)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    _build_ext.build_extension(self, ext)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/setuptools/_distutils/command/build_ext.py", line 548, in build_extension
    objects = self.compiler.compile(
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 525, in unix_wrap_ninja_compile
    cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 424, in unix_cuda_flags
    cflags + _get_cuda_arch_flags(cflags))
  File "/usr/wiss/lhao/anaconda3/envs/anygrasp2/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 1562, in _get_cuda_arch_flags
    arch_list[-1] += '+PTX'
IndexError: list index out of range
```
根据chatgpt提示
```
export TORCH_CUDA_ARCH_LIST="8.6"
python setup.py install
```
新报错
```

```

```
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI.git
```


# MyInstallation 2
atcremers95
```
conda create -n anygrasp python=3.9

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

pip install ninja

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

cd pointnet2
python setup.py install

pip install numpy Pillow scipy tqdm open3d tensorboard

pip install graspnetAPI

<!-- ## GraspNet api
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline

pip install tensorboard

cd knn
python setup.py install -->