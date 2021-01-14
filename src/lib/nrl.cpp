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

   Chame, H. F., Ahmadi, A., & Tani, J. (2020).
   A hybrid human-neurorobotics approach to primary intersubjectivity via
   active inference. Frontiers in psychology, 11.

   Okinawa Institute of Science and Technology Graduate University (OIST)
   Cognitive Neurorobotics Research Unit (CNRU)
   1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

-->*/

#include "LibNRL.h"

using namespace oist;

/**
 * This file is only wraps the API of LibNRL for shared library compilation
 * The documentation of these methods should be consulted in LibNRL.h
 *
 * */

extern "C" {


	/**
	 * Gets a singleton instance to a LibNRL object
	 * @return A pointer to a LibNRL instance
	 * */
	LibNRL* getInstance(){
		return LibNRL::getInstance();

	}

	/**
	 * Creates a new model
	 * @param nrl Pointer to a LibNRL instance
	 * @param path Full path to the properties file
	 * */
	void newModel(LibNRL* nrl, const char* path){
		nrl->newModel(string(path));
	}

	/**
	 * Loads the network model
	 * @param nrl Pointer to a LibNRL instance
	 * */
	void load(LibNRL* nrl){
		nrl->load();
	}

	/**
	 * Gets the number of degrees of freedom (DoF) of the network output
	 * @param nrl Pointer to a LibNRL instance
	 * @return Number of DoF
	 * */
	int getNDof(LibNRL* nrl){
		return nrl->getNDof();
	}

	/**
	 * Gets the number of intermediate layers
	 * @param nrl Pointer to a LibNRL instance
	 * @return Number of layers
	 * */
	int getNLayers(LibNRL* nrl){
		return nrl->getNLayers();
	}

	/**
	 * Gets the dimension of the network state space
	 * @param nrl Pointer to a LibNRL instance
	 * @return Dimension
	 * */
	int getStateDim(LibNRL* nrl){
		return nrl->getStateDim();
	}

	// -------------------------- training mode methods --------------------------

	/**
	 * Training in background mode. The training parameters are
	 * @param nrl Pointer to a LibNRL instance
	 * fully red from the properties file
	 * */
	void t_background(LibNRL* nrl){
		nrl->t_background();
	}

	/**
	 * Initialization of interactive training mode
	 * @param nrl Pointer to a LibNRL instance
	 * @param show Flag indicating to log training in the stdout
	 * */
	void t_init(LibNRL* nrl, bool show){
		nrl->t_init(show);
	}

	/**
	 * Loop iteration in interactive training mode
	 * @param nrl Pointer to a LibNRL instance
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
	void t_loop(LibNRL* nrl, float* output, int n){
		nrl->t_loop(output, n);
	}

	/**
	 * Ending of interactive training mode
	 * @param nrl Pointer to a LibNRL instance
	 * */
	void t_end(LibNRL* nrl){
		nrl->t_end();
	}

	// -------------------------- experiment mode methods ------------------------

	/**
	 * Enables on-line experiment mode
	 * @param nrl Pointer to a LibNRL instance
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
	void e_enable(LibNRL* nrl,
				int pID,
				int ws,
				float* param,
				int ne,
				int epoch,
				float alpha,
				float beta1,
				float beta2,
				bool store_s,
				bool store_p){

		nrl->e_enable(pID, ws, param, ne, epoch, alpha, beta1, beta2, store_s, store_p);
	}

	/**
	 * Computes the post-diction (inference) process
	 * @param nrl Pointer to a LibNRL instance
	 * @param input Sliding window buffer array
	 * @param output Array for loss function components (reconstruction error, regulation error, loss)
	 * @param show A flag indicating to show log in the standard output
	 * */
	void e_postdict(LibNRL* nrl, float* input, float* output, bool show){

		return nrl->e_postdict(input, output, show);
	}

	/**
	 * Generates output from the prior distribution
	 * @param nrl Pointer to a LibNRL instance
	 * @param output Array for storing the robot joint positions
	 * */
	void e_generate(LibNRL* nrl, float* output){

		nrl->e_generate(output);
	}

	/**
	 * Stores the experimental data in disk in case this functionality was set in @ref e_enable
	 * It is recommended saving data in background only for short-time experiments,
	 * since performance can be affected. Another way of storing information is through the method @ref e_getState
	 * which do not buffer data history but provides only the current state.
	 * Hence, the client application is in charge of storing data locally.
	 * @param nrl Pointer to a LibNRL instance
	 * @param path Output full path
	 * */
	void e_save(LibNRL* nrl, const char* path){

		nrl->e_save(path);
	}

	/**
	 * Gets the current state of the network and writes data to the output buffer.
	 * This method is recommended for saving data in experiments.
	 * @param nrl Pointer to a LibNRL instance
	 * @param output Array with latent state, whose dimension can be obtained through *getStateDim* once the model has been loaded.
	 * */
	void e_getState(LibNRL* nrl, float* output){
		nrl->e_getState(output);
	}

	// -------------------------- off-line mode methods --------------------------

	/**
	 * Generative off-line n time steps predictions from the initial context provided and store computations on disk.
	 * @param nrl Pointer to a LibNRL instance
	 * @param path Output full path dir
	 * @param n Number of time steps
	 * @param input Array with latent state, whose dimension can be obtained through 'getStateDim'
	 * */
	void a_predict(LibNRL* nrl, const char* path, int n, float* input){
		nrl->a_predict(path, n, input);
	}

	/**
	 * Computes the output from feed-forwarding a given context.
	 * This method does not modify the internal state of the network,
	 * (i.e. the generative and inference processes are not involved).
	 * It only computes the output layer from a given context.
	 * @param nrl Pointer to a LibNRL instance
	 * @param input Array with latent state, whose dimension can be obtained through 'getStateDim'
	 * @param output Array container
	 * */
	void a_getOutputFromContext(LibNRL* nrl, float* input, float* output){
		nrl->a_feedForwardOutputFromContext(input, output);
	}

}



