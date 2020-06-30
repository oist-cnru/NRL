# Neural Robotics Library (NRL)

The *Neural Robot Library* (NRL) is an implementation in the C++ programming language of a neuro-cognitive control scheme for human-robot interaction experiments. The default neural network type included is a *Predictive-coding-inspired Variational Recurrent Neural Network* (PV-RNN) architecture.

## Project content

In this repository the implementation of NRL is provided. A stand-alone demonstration program (*src/standalone/main.cpp*) is proposed to illustrate how to train and perform on-line inference with NRL. The sources can also be compiled as a shared library and imported to a Python version 3 programming environment. The tutorial video below presents the project and explains how to build and run the demonstration programs.

<a href="https://youtu.be/bVgZstMjDWM" rel="some text">![](images/tutorial1.png?raw=true)</a>

An interesting NRL application program for real-time interaction with a virtual Cartesian robot in the Python 3 language is available in the [VCBot project](https://github.com/oist-cnru/VCBot.git). 

## Software design

The software engineering project aimed at achieving three important objectives: a) to obtain real-time performance, b) to preserve the mathematical structure of neural network models as much as possible, and c) to comply with important software engineering principles, such as readability, maintainability, and extensibility. For this, the Object Oriented programming paradigm was selected for the project. The figure below shows the simplified class diagram of NRL. The full source documentation is available. After downloading the sources, you just have to open the file ![documentation.html](documentation.html) locally in your browser.

<!-- for doxygen -->
<!-- ![](@ref images/classdiagram.png) -->
<!-- for github -->
![](images/classdiagram.png?raw=true)

Three interfaces (abstract classes) are provided, to offer a certain level of abstraction in the source. These classes are: *IRobot*, *INetwork*, and *ILayer*. Hence, the inclusion of a new robot configuration, for instance, would require the new class inheriting and implementing the virtual methods in *IRobot*. By exploiting the property of polymorphism, minor changes should be done in other parts of the sources. An analogous principle should be applied for creating networks with diverse sorts of layers. Finally, in order to ensure computation efficiency, network and layer objects share data through context objects, which inherit from the interface *IContext*.


## The operation modes

There are three operation modes available is NRL:

- **Training Mode**
 
  In this mode, the model is provided with a data-set for training. The methods in the Application Program Interface (API) related to this mode are denoted starting by the prefix *t_*. There are two ways of training: a) in background, b) interactively. When training in background (*LibNRL::t_background*), the client program waits for the whole model to be trained before regaining control in the application. For input/output efficiency, in NRL training is persisted in permanent storage each *min(nEpochs, nEpochs modulus 100)* epochs. In case of training interactively (*LibNRL::t_init*, *LibNRL::t_loop*, *LibNRL::t_end*), it is possible to regain control after each time data is saved. This is convenient for graphical user interface (GUI) based application clients, offering the possibility to cancel the training process.

- **Experiment Mode**
 
  This mode is for real-time interaction with the robot. The related methods are denoted starting by the prefix *e_* in the API. It is required to enable the mode (*LibNRL::e_enable*) before performing experiments. Other useful methods are *LibNRL::e_generate* for behavior generation and *LibNRL::e_postdict* for on-line inference. The description of the methods signature is documented in the source files.

- **Analysis Mode**

  Sometimes it is convenient to compute the output of the network from recorded states. These functionalities are performed off-line, and serve to analytical purposes. Therefore, the *analysis* methods are denoted starting by the prefix *a_* in the API.
 

## Requirements

The project has been successfully tested in the Ubuntu, OS X, and Windows operating systems.

It is required the *Eigen C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms*. The version 3.3.0 has been tested in the project. Please choose a version for download from [here](https://gitlab.com/libeigen/eigen/-/releases)

The *C++ Standard Template Library (STL)* is also required. Please choose the implementation of your preference. The GNU implementation has been successfully tested. For Ubuntu, this compiler is pre-installed with the distribution. For Windows it is recommended the projects [MinGW](http://www.mingw.org/) for 32 Bits platforms, and [CygWin](https://www.cygwin.com/) for 64 Bits platforms.

For building the sources it is recommended to install CMAKE. The instructions on how to install this tool in your platform are provided [here](https://cmake.org/install/).
 
### Compiling NRL

The first step to do is providing CMAKE with the path to the parent directory containing the folder *Eigen*. For this, please change the line 54 (*include_directories(/PATH_TO_FOLDER_EIGEN)* of *src/standalone/CMakeLists.txt* (for the stand-alone program) or *src/lib/CMakeLists.txt* for compiling the shared library.

- **The stand alone demonstration program**

  Create a folder to build the sources and address the route to the folder *src/standalone* to CMAKE. For example, if the folder is to be created in the projects' root directory, you should proceed analogous to this:  

```
mkdir build
cd build
cmake ../src/standanole
make
```


- **The shared library**

  Similar to the previous case, just change the path to the relevant *CMakeLists.txt* file  

```
mkdir build
cd build
cmake ../src/lib
make
```
  A wrapper class for NRL in Python version 3 is provided in 'NRL/python/NRL.py'. Thus, the same functionalities provided in the stand-alone program are available in Python 3. By default, 'NRL.py' searches for the shared library in the folder 'NRL/python/lib'. You can proceed either by creating this directory and compiling the library there, or, by editing the file 'NRL.py' and changing the path to the shared lib in the variable 'libFolder'.

## Instructions

 To run the stand-alone demo it is required to provide two arguments:
```
./NRL_SA PATH_TO_PROJECT/data/properties_ROBOT.d [train|sim]
```
 For running the python version of the stand-alone demo ('NRL/python/main.py'), you should proceed as follows:
```
python3 main.py PATH_TO_PROJECT/data/properties_ROBOT.d [train|sim]
```
 
There are three kind of robots available by default in NRL:

- Robot Cartesian (3 degrees of freedom )
- Robot Torobo (16 degrees of freedom )
- Robot Generic (28 degrees of freedom )

 In case the configuration of these robots do not fit your project, you could create a new robot class in the *src/robot* folder. Alternatively, you could edit an existent class (for instance the *src/robot/Generic.h* and *src/robot/Generic.cpp*) to match the number of joints and the information on the joint limits of your project.

 Finally, as explained below, the parameters for the neural network models are set in a dedicated properties file. Some examples are provided in the folder *data/config*.

 ### Demonstration

 A dummy dataset is provided to analyze the computational performance in your system. The dataset is composed of sequences of 100 time steps with random floating point data. The primitives in *data/dataset* are available in the human-readable file format *comma-separated value* (CSV). The naming convention followed is *primitive_ID_SAMPLING.csv*. Here *ID* is an integer identification number assigned to the primitive, and *SAMPLING* is an integer denoting the number of samples available for the primitive. In total, three primitives conform the dataset (*ID*=0: 1 sample, *ID*=1: 3 samples, *ID*=2: 2 samples).    


 ### Results

 The following results were obtained running the stand-alone program. The host platform was a computer Alienware Aurora R7, 12 Intel® CoreTM i7-8700K CPU at 3.70GHz, and 32 GiB RAM memory, running in Ubuntu 16.04 LTS. When running the version built as a shared library and included in Python 3, the performance is estimated to drop around 30%.
 
 - **The robot Cartesian**
 
  *Training*
```
Training from scratch ...

Epoch [100] - Time [23487.6ms] - RE_Q [335.409] - RE_P [653.864] - Regulation [39.4366] - loss [119.69]
Model saved!
Epoch [200] - Time [23476ms] - RE_Q [308.124] - RE_P [371.773] - Regulation [21.2025] - loss [106.949]
Model saved!
Epoch [300] - Time [23503.5ms] - RE_Q [304.847] - RE_P [308.088] - Regulation [12.4978] - loss [104.115]
Model saved!
Epoch [400] - Time [23452.8ms] - RE_Q [295.217] - RE_P [317.019] - Regulation [8.40558] - loss [100.087]
Model saved!
Epoch [500] - Time [23461.4ms] - RE_Q [302.422] - RE_P [294.137] - Regulation [5.31455] - loss [101.87]
Epoch [600] - Time [23455.1ms] - RE_Q [294.582] - RE_P [314.82] - Regulation [4.38003] - loss [99.0701]
Model saved!
Epoch [700] - Time [23456ms] - RE_Q [288.958] - RE_P [293.106] - Regulation [3.92122] - loss [97.1034]
Model saved!
Epoch [800] - Time [23514.9ms] - RE_Q [303.388] - RE_P [288.831] - Regulation [4.38013] - loss [102.005]
Epoch [900] - Time [23526.6ms] - RE_Q [288.442] - RE_P [293.512] - Regulation [3.33614] - loss [96.8146]
Model saved!
Epoch [1000] - Time [23661.8ms] - RE_Q [289.363] - RE_P [287.358] - Regulation [2.90963] - loss [97.0364]

Training end

```
 *Simulation*
```
simulation demonstration begin

Model loaded!
Step: 1 in 43.1785 ms
Step: 2 in 43.2941 ms
Step: 3 in 43.0191 ms
Step: 4 in 42.9615 ms
Step: 5 in 43.2043 ms
Step: 6 in 43.0813 ms
Step: 7 in 42.9529 ms
Step: 8 in 43.2127 ms
Step: 9 in 43.0466 ms
Step: 10 in 43.0158 ms
Step: 11 in 43.24 ms
Step: 12 in 42.967 ms
Step: 13 in 42.986 ms
Step: 14 in 43.2415 ms
Step: 15 in 42.996 ms

simulation demonstration end

```

 - **The robot Torobo**
 
  *Training*
```
Training from scratch ...

Epoch [100] - Time [55993.1ms] - RE_Q [1188.61] - RE_P [3420.26] - Regulation [49.8778] - loss [84.2634]
Model saved!
Epoch [200] - Time [55829.3ms] - RE_Q [948.287] - RE_P [1551.03] - Regulation [29.44] - loss [65.1559]
Model saved!
Epoch [300] - Time [55950.4ms] - RE_Q [879.619] - RE_P [824.855] - Regulation [21.3711] - loss [59.2504]
Model saved!
Epoch [400] - Time [55894.8ms] - RE_Q [799.868] - RE_P [1551.69] - Regulation [12.2354] - loss [52.4389]
Model saved!
Epoch [500] - Time [55822.1ms] - RE_Q [852.259] - RE_P [787.73] - Regulation [9.52194] - loss [55.1706]
Epoch [600] - Time [55900.4ms] - RE_Q [785.516] - RE_P [1027.18] - Regulation [6.89619] - loss [50.474]
Model saved!
Epoch [700] - Time [55840.3ms] - RE_Q [714.994] - RE_P [772.807] - Regulation [5.43834] - loss [45.7748]
Model saved!
Epoch [800] - Time [55585.8ms] - RE_Q [934.616] - RE_P [721.111] - Regulation [7.83391] - loss [59.9803]
Epoch [900] - Time [55570ms] - RE_Q [712.984] - RE_P [823.665] - Regulation [4.65715] - loss [45.4929]
Model saved!
Epoch [1000] - Time [55929.5ms] - RE_Q [735.673] - RE_P [762.851] - Regulation [4.19637] - loss [46.8189]

Training end

```
 *Simulation*
```
simulation demonstration begin

Model loaded!
Step: 1 in 93.0724 ms
Step: 2 in 91.6953 ms
Step: 3 in 91.4821 ms
Step: 4 in 91.3739 ms
Step: 5 in 91.3501 ms
Step: 6 in 90.0752 ms
Step: 7 in 90.2209 ms
Step: 8 in 90.3339 ms
Step: 9 in 90.9228 ms
Step: 10 in 91.0096 ms
Step: 11 in 90.8634 ms
Step: 12 in 90.0855 ms
Step: 13 in 90.6236 ms
Step: 14 in 90.3553 ms
Step: 15 in 90.2141 ms

simulation demonstration end

```

 - **The robot Generic**
 
  *Training*
```
Training from scratch ...

Epoch [100] - Time [86071.6ms] - RE_Q [1817.14] - RE_P [6922.24] - Regulation [31.7749] - loss [71.2527]
Model saved!
Epoch [200] - Time [85936.5ms] - RE_Q [1222.26] - RE_P [2633.2] - Regulation [20.5835] - loss [47.7688]
Model saved!
Epoch [300] - Time [85904.5ms] - RE_Q [1246.48] - RE_P [1147.18] - Regulation [12.1075] - loss [46.9388]
Model saved!
Epoch [400] - Time [85831.3ms] - RE_Q [1098.29] - RE_P [1140.75] - Regulation [4.81528] - loss [40.1878]
Model saved!
Epoch [500] - Time [85906ms] - RE_Q [1132.96] - RE_P [1084.95] - Regulation [4.67464] - loss [41.3977]
Epoch [600] - Time [86025.3ms] - RE_Q [1059.51] - RE_P [1249.31] - Regulation [3.55755] - loss [38.5511]
Model saved!
Epoch [700] - Time [86144.7ms] - RE_Q [1023.01] - RE_P [1046.9] - Regulation [3.29288] - loss [37.1945]
Model saved!
Epoch [800] - Time [85945.3ms] - RE_Q [1150.83] - RE_P [1012.85] - Regulation [3.54976] - loss [41.8111]
Epoch [900] - Time [85829.8ms] - RE_Q [1023.06] - RE_P [1096.09] - Regulation [3.23186] - loss [37.1844]
Model saved!
Epoch [1000] - Time [85667.2ms] - RE_Q [1035.68] - RE_P [1001.85] - Regulation [2.6702] - loss [37.5226]

Training end
```
 *Simulation*
```
simulation demonstration begin

Model loaded!
Step: 1 in 134.842 ms
Step: 2 in 134.769 ms
Step: 3 in 134.787 ms
Step: 4 in 134.858 ms
Step: 5 in 134.903 ms
Step: 6 in 134.898 ms
Step: 7 in 134.867 ms
Step: 8 in 134.838 ms
Step: 9 in 134.846 ms
Step: 10 in 134.804 ms
Step: 11 in 134.819 ms
Step: 12 in 134.899 ms
Step: 13 in 134.856 ms
Step: 14 in 134.909 ms
Step: 15 in 134.937 ms

simulation demonstration end
```

 
## Backend parametrization {#backendParams}

NRL loads a configuration file specifying the parameters of the model. Some of these parameters ensure the operation of NRL, others are specific to the network architecture contained in the library (e.g. the parameters *w*, *d*, *z*, and *t* are specific to the neural network type PV-RNN). The *delimiter* char for separating multiple fields is the comma (','). The fields described next are separated from values by the character '='.
 
|Parameter|Description|
|---------|-----------|
|datapath |Full path to the data-set |
|modelpath|Full path to the model|
|robot|The robot name (e.g ‘cartesian’,’torobo’, ‘generic’)|
|activejoints| A *delimiter* separated integer 1 (active) or 0 (Inactive), indicating the joint's activation (e.g. '1,1,0' the first two out of three joints are activated)|
|nsamples| Integer number of samples per primitive separated by *delimiter* (e.g. '1,1,1' for three primitives with one sample each)|
|w|PV-RNN: Real numbers for meta parameters *w* per layer for training the model (e.g '0.025,0.025' for a two-layered network)|
|d|PV-RNN: Integer number of *d* units per layer for training the model (e.g. '40,10' indicating 40 units in layer one and 10 units in layer two)|
|z|PV-RNN: Integer number of *z* units per layer for training the model (e.g. '4,1' indicating 4 units in layer one and 1 units in layer two)|
|t|PV-RNN: Integer time constants per layer for training the model (e.g. '2,10' indicating 2 time steps for layer one and 10 time steps for layer two)|
|epochs|Integer number of training epochs (e.g. '50000')|
|alpha|Adam optimization parameter &alpha; for training the model (e.g. '0.001')|
|beta1|Adam optimization parameter &beta;<sub>1</sub> for training the model (e.g. '0.9')|
|beta2|Adam optimization parameter &beta;<sub>2</sub> for training the model (e.g. '0.999')|
|shuffle|Boolean flag indicating to shuffle the training order of the primitives in the dataset (e.g. 'true' or 'false')|
|retrain|Boolean flag indicating the intention to retrain the model, or overwrite previous training (e.g. 'true' or 'false')|
|greedy|Boolean flag indicating the intention to save the model only if optimal loss function value was obtained (e.g. 'true' or 'false')|
|dsoft|Real number indicating the distance between reference values in the joint space encoded by the softmax function (e.g. '10' would indicate a neuron for each 10 units in the joint space)|
|sigma|Real number for the sigma parameter in the softmax function (e.g. '0.2')|

## Variable naming convention

In the table below it is presented the variable naming convention adopted in order to improve the readability of the sources.

|Token|Description|
|---------|-----------|
|t|training mode|
|e|experiment mode|
|c|context|
|g|gradient|
|m|mean (first moment, ADAM optimization)|
|v|uncentered variance (second moment, ADAM optimization)|
|W|synaptic weight matrix (e.g 'Wdh' means synaptic weights from the latent state 'd' to the latent state 'h')|
|B|bias|
|p|prior distribution|
|q|posterior distribution|
|d|latent state d|
|h|latent state h|
|n|noise|
|z|latent state z|
|u|latent state mu|
|s|latent state sigma|
|l|latent state log sigma|
|w|meta-parameter w|
|i|iterator|
|au|parameter a mu|
|al|parameter a log sigma|
|ut|utils|
|id|identity|
|bw|backward computation|
|kld|Kullback-Leibler divergence|
|bottom|bottom level|
|top|top level|
|gen|variable associated with the generation process|
|opt|optimal|
|cur|current|
|dim|dimension|
|num|number|
|sum|summation|
|sub|subtraction|
|div|division|
|mul|multiplication|
|pow|power|
|len|length|
|tau|Greek letter tau|
|eps|Greek letter epsilon|
|rec|reconstruction component of the loss function|
|reg|regulation component of the loss function|
|coef|coefficient|
|prim|primitive|
|loss|loss function|
|next|time t = t + 1|
|prev|time t = t - 1|
|tanh|hyperbolic tangent|
|tzero|time t = 0|
|thres|threshold|
|transpose|transpose of a vector or matrix|


## Work reference

H. F. Chame, A. Ahmadi, J. Jun (2020) *Towards hybrid primary intersubjectivity: a neural robotics library for human science*


## Contact

This program was implemented by: Hendry F. Chame

**Lab** Cognitive Neurorobotics Research Unit (CNRU)

**Institution** Okinawa Institute of Science and Technology Graduate University (OIST)

**Address** 1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

**E-mail** hendryfchame@gmail.com

