# SLR-NTF
Spatial Low Rank Nonnegative Tensor Factorization

Installing Tensorflow for Windows
=================================
1. Download miniconda
2. Run a miniconda prompt
3. conda create -n tf
4. conda activate tf
5. conda install tensorflow
    - This will install tensorflow 2.3 and all required
    dependencies at the correct versions including a 
    python environment, intel mkl libraries, numpy, etc.
6. conda install matplotlib
7. At this point you can run the tensor factorizations
  for the included datasets: h01-samson, h02-jasper, and
  h03-urban by just typing: python runall.py.  It will run
  in parallel on a multicore CPU.

Installing GPU Support
======================
In order to take advantage on a GPU you need to install
tensorflow-gpu with tensorflow 2.0 or greater.  Only Nvidia
GPUs are supported and you will need the correct version
of the cuda toolkit and driver.
If Nvidia libraries or the CUDA runtime is not properly setup
tensorflow will default to running on the CPU.  If a GPU is
detected, tensorflow will load the appropriate library and
use it.
8. conda create -n tf-gpu
9. conda activate tf-gpu
10. conda install tensorflow-gpu
11. conda install matplotlib

Installing Jupyter Notebook:
============================
12. On either environment run: conda install jupyter
13. A shortcut is installed on the Start menu that will
launch the Jupyter Server and a browser screen pointing
to it at: http://loacalhost:8888
14. Browse for demo.ipynb
15. Click on Kernel->Restart and Run All
