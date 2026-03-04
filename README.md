# boe-step-4-vertical-displacement

Include this pre-trained u-net model 'unet_membrane.hdf5' in the root of this project: 
https://drive.google.com/file/d/1Kjf2_7WaVH_Xq6RZ08otZ8j5BE6068cF/view?usp=drive_link

We recommend the following versions for the required dependencies in your virtual environment or anaconda environment:
- Python 3.10
- Tensorflow 2.10
- Keras 2.10
- Numpy 1.26.4
- Matplot 3.8.4
- Pandas 2.1.4
- Pillow 11.2.1


The Pointcloud2Orthoimage.py has the following version requirements which should be installed alongside the previously mentioned versions:
- Laspy 2.5.4
- Open3d 0.17.0
- OpenCV 4.5.5
- Scipy 1.11.4

Not Recommended but up to experimentation: You may use newer version of Tensorflow such as 2.12, but upgrading python to 3.11 is necessary. This will require newer versions of OpenCV:
- Tensorflow 2.12
- Keras 2.12


If not using Anaconda, create virtual environment using the following commands:
1. First ensure that you have python 3.10 installed
2. Create a virtual environment using: python3.10 -m venv venv
3. Run this command to activate: source venv/bin/activate
4. Change the pointcloud file path in main
5. Do python main.py to execute
