/*<!--

 BSD 3-Clause License

  Copyright (c) 2020 Okinawa Institute of Science and Technology (OIST).
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
   * Neither the name of Willow Garage, Inc. nor the names of its
     contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.

 Author: Hendry F. Chame <hendryfchame@gmail.com>

 Publication:

   "Towards hybrid primary intersubjectivity: a neural robotics 
   library for human science"

   Hendry F. Chame, Ahmadreza Ahmadi, Jun Tani

   Okinawa Institute of Science and Technology Graduate University (OIST)
   Cognitive Neurorobotics Research Unit (CNRU)
   1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

-->*/


#ifndef LIB_LIBNRL_H_
#define LIB_LIBNRL_H_

#include "../includes.h"
#include "../utils/Utils.h"

#include "../robot/Cartesian.h"
#include "../robot/Torobo.h"
#include "../robot/Generic.h"

#include "../dataset/Dataset.h"
#include "../network/INetwork.h"
#include "../network/NetworkPvrnn.h"
#include "../network/NetworkPvrnnBeta.h"


namespace oist {

/**
 * This class provides the NRL functionalities for training a neural network model,
 * performing on-line experiments, and computing off-line analysis.
 * For this, the class considers the parameters provided in a properties file to the method @ref newModel.
 * The description of the parameters can be found in the document section [Backend parametrization](@ref README.md)
 * The class implements a singleton design pattern.
 *
 * */
class LibNRL {

	Dataset* dataset;
	INetwork* model;
	ofstream* logFile;
	IRobot* robot;
	Utils* ut;

	string modelPath;
	string dataPath;
	string robotName;
	string networkName;
	string strEpoch;
	string propPath;
	string modelNUllMsg;
	bool stdoutLog;

	const string dataPrefix;

	// softmax primitives in the data-set
	vectorXf4DContainer YSoftmax;

	float nDof;
	int nSeq;
	int seqLen;

	bool loaded;
	float dsoft;
	float sigma;
	float maxLoss;

	bool1DContainer activeJoints;
	int1DContainer nSamples;

	// variables for training mode
	int t_nEpoch;
	int t_step;
	float t_alpha;
	float t_beta1;
	float t_beta2;
	bool t_shuffle;
	bool t_retrain;
	bool t_greedy;
	float1DContainer t_w;
	int1DContainer t_prim_Ids;

	// variables for experiment mode
	int e_winSize;
	int e_nEpoch;
	int e_step;
	float e_alpha;
	float e_beta1;
	float e_beta2;
	float1DContainer e_w;

	static LibNRL* myInstance;

	/**
	 * Constructor
	 * */
	LibNRL();


	/**
	 * Destructor
	 * */
	~LibNRL();

	/**
	 * Deallocates resources
	 * */
	void deallocate();

public:

	static LibNRL* getInstance();

	/**
	 * Creates a new model
	 * @param path Property file full path
	 * */
	void newModel(string path);

	/**
	 * Gets the number of degrees of freedom (DoF) of the network output
	 * @return Number of DoF
	 * */
	int getNDof();

	/**
	 * Gets the number of intermediate layers
	 * @return Number of layers
	 * */
	int getNLayers();

	/**
	 * Gets the dimension of the network state space
	 * @return Dimension
	 * */	
	int getStateDim();
	
	/**
	 * Loads the network model
	 * */	
	void load();

	// -------------------------- training mode --------------------------

	/**
	 * Initialization of interactive training mode
	 * @param show Flag indicating to log training in the stdout
	 * */	
	void t_init(bool show);


	/**
	 * Loop iteration in interactive training mode
	 * @param output Output array for the tuple (step, RE_Q, RE_P, REG, loss, opt)
	 *    Field  | Description
	 *    ------------- | -------------
	 *    step  | Current training time step
	 *    RE_Q  | Reconstruction error from the posterior distribution
	 *    RE_P  | Reconstruction error from the prior distribution
	 *    REG   | Regulation error KL-Div(Prior,posterior)
	 *    loss  | Loss function error
	 *    opt   | Integer flag: (1) Global optimal loss, (0) otherwise
	 * @param n Number of training epochs
	 * */	
	void t_loop(float* output, int n);

	/**
	 * Ending of interactive training mode
	 * */
	void t_end();

	/**
	 * Training in background mode. The training parameters are
	 * fully red from the properties file
	 * */
	void t_background();


	// -------------------------- experiment mode ------------------------

	/**
	 * Enables on-line experiment mode
	 * @param pID Primitive ID
	 * @param ws Sliding window size
	 * @param param Parameters array for the neural network model
	 * @param ne Number of experiment time steps
	 * @param epoch Number of post-diction epochs
	 * @param alpha Adam optimization hyper parameter \f$\alpha\f$
	 * @param beta1 Adam optimization hyper parameter \f$\beta_1\f$
	 * @param beta2 Adam optimization hyper parameter \f$\beta_2\f$
	 * @param store_s A flag indicating to store the states on disk (if true performance can be affected)
	 * @param store_p A flag indicating to store intermediate BPPT computations (if true performance can be affected)
	 * */
	void e_enable(int pID, int ws, float* param, int ne, int epoch, float alpha, float beta1, float beta2, bool store_s, bool store_p);

	/**
	 * Computes the post-diction (inference) process
	 * @param input Sliding window buffer array
	 * @param output Array for loss function components (reconstruction error, regulation error, loss)
	 * @param show A flag indicating to show log in the standard output
	 * */
	void e_postdict(float* input, float* output, bool show);

	/**
	 * Generates output from the prior distribution
	 * @param output Array for storing the robot joint positions
	 * */
	void e_generate(float* output);

	/**
	 * Stores the experimental data in disk in case this functionality was set in @ref e_enable
	 * It is recommended saving data in background only for short-time experiments,
	 * since performance can be affected. Another way of storing information is through the method @ref e_getState
	 * which do not buffer data history but provides only the current state.
	 * Hence, the client application is in charge of storing data locally.
	 * @param path Output full path
	 * */
	void e_save(string path);

	/**
	 * Gets the current state of the network and writes data to the output buffer.
	 * This method is recommended for saving data in experiments.
	 * @param output Array with latent state, whose dimension can be obtained through *getStateDim* once the model has been loaded.
	 * */
	void e_getState(float* output);

	// -------------------------- analysis mode --------------------------
	/**
	 * Generative off-line n time steps predictions from the initial context provided and store computations on disk.
	 * @param path Output full path dir
	 * @param n Number of time steps
	 * @param input Array with latent state, whose dimension can be obtained through 'getStateDim'
	 * */
	void a_predict(const char* path, int n, float* input);


	/**
	 * Computes the output from feed-forwarding a given context.
	 * This method does not modify the internal state of the network,
	 * (i.e. the generative and inference processes are not involved).
	 * It only computes the output layer from a given context.
	 * @param input Array with latent state, whose dimension can be obtained through 'getStateDim'
	 * @param output Array container
	 * */
	void a_feedForwardOutputFromContext(float* input, float* output);

};

} /* namespace oist */

#endif /* LIB_LIBNRL_H_ */
