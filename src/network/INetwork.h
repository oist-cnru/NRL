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

#ifndef NETWORK_INETWORK_H_
#define NETWORK_INETWORK_H_

#include "../includes.h"

namespace oist {

/**
 * Abstract class (Interface) for network implementations
 * */
class INetwork {

public:

	/**
	 * Get the number of layers
	 * @return An integer representing the number of layers
	 * */
	virtual int getNLayers() = 0;

	/**
	 * Gets the dimension of the network state space
	 * @return An integer representing the dimension
	 * */
	virtual int getStateDim() = 0;

	/**
	 * Load the network model
	 * @param path Model directory full path
	 * */
	virtual void load(string path) = 0;

	/**
	 * Save the network model
	 * @param path Model directory full path
	 * */
	virtual void save(string path) = 0;

	/**
	 * Get the reconstruction error
	 * @param gen Container with the output generation by network
	 * @param ref Container with the reference data
	 * @return A real value for the reconstruction error
	 * */
	virtual float getRecError(vectorXf2DContainer& gen, vectorXf3DContainer& ref) = 0;

	// ------------------------- training mode methods -------------------------

	/**
	 * *[Training mode]* Generation from the prior distribution
	 * @param n number of time steps
	 * @param pID Primitive ID
	 * @param output Container for output recording
	 * */
	virtual void t_generate(int n, int pID, vectorXf2DContainer& output) = 0;

	/**
	 * *[Training mode]* Generation from the posterior distribution
	 * @param n number of time steps
	 * @param pID Primitive ID
	 * @param output Container for output recording
	 * */
	virtual void t_forward(int n, int pID, vectorXf2DContainer& output) = 0;

	/**
	 * *[Training mode]* Back propagation through time computation (inference)
	 * @param epoch current epoch number
	 * @param X Input container with network generation from the posterior distribution @ref t_forward
	 * @param Y Input container with the reference data
	 * @param rec Output reconstruction error
	 * @param reg Output regulation error
	 * @param loss Output loss function
	 * */
	virtual void t_backward(int epoch, vectorXf2DContainer& X, vectorXf3DContainer& Y, float& rec, float& reg, float& loss) = 0;

	/**
	 * *[Training mode]* Computes ADAM optimization of parameters
	 * @param pID Primitive ID
	 * @param alpha Adam optimization hyper parameter \f$\alpha\f$
	 * @param beta1 Adam optimization hyper parameter \f$\beta_1\f$
	 * @param beta2 Adam optimization hyper parameter \f$\beta_2\f$
	 * */
	virtual void t_optAdam(int pID, float alpha, float beta1, float beta2) = 0;

	// ------------------------- Analysis mode methods -------------------------


	/**
	 * *[Analysis mode]* Computes the generative process off-line and stores on disk
	 * @param n Number of generations
	 * @param input Input array with initial state for all the layers concatenated
	 * @param path Full path directory to save the output
	 * */
	virtual void a_predict(int n, float* input, string path) = 0;

	/**
	 * *[Analysis mode]* Computes the output from feed-forwarding a given context.
	 * This method does not modify the internal state of the network,
	 * (i.e. the generative and inference processes are not involved).
	 * It only computes the output layer from a given context.
	 * @param input  Input array with latent state for the involved layers concatenated
	 * @param output Output container
	 * */
	virtual void a_feedForwardOutputFromContext(float* input, float* output) = 0;

	// ------------------------- Experiment mode methods -------------------------

	/**
	 * *[Experiment mode]* Enables the experiment mode
	 * @param pID Primitive ID
	 * @param winSize Sliding window size
	 * @param param container with parameters
	 * @param nT Number of experiment times
	 * @param store_gen A flag indicating to store the generative process states. This should be used with caution,
	 *     since memory allocation for large data may degrade performance. It is recommended to use @ref e_getState
	 *     to get the current state of the network, without storing information on the back-end side.
	 *     Hence, the client front-end does the storage.
	 * @param store_inf A flag indicating to store the inference process states. Valuable only if the parameter *store_gen*
	 *     is set true. Thus should be used only for debugging, since memory allocation for large data may degrade performance.
	 *     All the computation steps for the *back propagation trough time* (BPPT) algorithm are stored.
	 * */

	virtual void e_enable(int pID, int winSize, float* param, int nT, bool store_gen, bool store_inf) = 0;

	/**
	 * *[Experiment mode]* Generation from the prior distribution
	 * @param output Container for output recording
	 * */
	virtual void e_generate(float* output) = 0;

	/**
	 * *[Experiment mode]* Initialize the inference process forward computations in experiment mode
	 * @return A flag indicating the success of the operation
	 * */
	virtual bool e_initForward() = 0;

	/**
	 * *[Experiment mode]* Generation from the posterior distribution
	 * @param output Container for output recording
	 * */
	virtual void e_forward(vectorXf2DContainer& output) = 0;

	/**
	 * *[Experiment mode]* Back propagation through time computation (inference)
	 * @param X Input container with network generation from the posterior distribution @ref e_forward
	 * @param Y Input container with the reference data
	 * @param rec Output reconstruction error
	 * @param reg Output regulation error
	 * @param loss Output loss function
	 * */
	virtual void e_backward(vectorXf2DContainer& X, vectorXf2DContainer& Y, float& rec, float& reg, float& loss) = 0;

	/**
	 * *[Experiment mode]* Copies the network's parameters. It is used for greedy optimization
	 * */
	virtual void e_copyParam() = 0;

	/**
	 * *[Experiment mode]* Overrides the network's parameters. It is used for greedy optimization
	 * */
	virtual void e_overwriteParam() = 0;

	/**
	 * *[Experiment mode]* Computes ADAM optimization of parameters
	 * @param pID Primitive ID
	 * @param alpha Adam optimization hyper parameter \f$\alpha\f$
	 * @param beta1 Adam optimization hyper parameter \f$\beta_1\f$
	 * @param beta2 Adam optimization hyper parameter \f$\beta_2\f$
	 * */
	virtual void e_optAdam(int pID, float alpha, float beta1, float beta2) = 0;

	/**
	 * *[Experiment mode]* Writes the current network state to a float array.
	 * Both the generation and inference latent states are provided real time for analysis.
	 * @param output Float array reference. The size of the array can be obtained from @ref getStateDim
	 * */
	virtual void e_getState(float* output) = 0;

	/**
	 * *[Experiment mode]* Save the network model
	 * @param path Output directory full path
	 * */
	virtual void e_save(string path) = 0;

	/**
	 * Destructor
	 * */
	virtual ~INetwork(){}



};

} /* namespace oist */

#endif /* NETWORK_INETWORK_H_ */
