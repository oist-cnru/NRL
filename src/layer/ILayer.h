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

#ifndef NETWORK_LAYER_ILAYER_H_
#define NETWORK_LAYER_ILAYER_H_

#include "../includes.h"
#include "../context/IContext.h"

namespace oist {

/**
 * Abstract class (Interface) for layer implementations
 * */
class ILayer {

public:

	/**
	 * Gets the dimension of the layer state space
	 * @return dimension
	 * */
	virtual int getStateDim() = 0;


	/**
	 * Load the layer's parameters
	 * @param path path Data full path
	 * */
	virtual void load(string path) = 0;

	/**
	 * Save the layer's parameters
	 * @param path Destination full path
	 * */
	virtual void save(string path) = 0;

	// ------------------------- context methods -------------------------

	/**
	 * Initialized the layer context
	 * @param pID Primitive ID
	 * */
	virtual void initContext(int pID) = 0;

	/**
	 * Gets the layer context
	 * */
	virtual IContext* getContext() = 0;

	// ------------------------- training methods -------------------------

	/**
	 * *[Training mode]* Computes the generative process
	 * @param time Current time step
	 * @param pID Primitive ID
	 * */
	virtual void t_generate(int time, int pID) = 0;

	/**
	 * *[Training mode]* Computes the forward inference process
	 * @param time Current time step
	 * @param pID Primitive ID
	 * */
	virtual void t_forward(int time, int pID) = 0;

	/**
	 * *[Training mode]* Initializes the inference process backward computation
	 * */
	virtual void t_initBackward() = 0;

	/**
	 * *[Training mode]* Computes the backward (BPTT algorithm) inference process
	 * @param time Current time step
	 * @param pID Primitive ID
	 * */
	virtual void t_backward(int time, int pID) = 0;

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
	 * *[Analysis mode]* Initialize off-line predictions
	 * @param: Input state at time t=0
	 * */
	virtual void a_init(float*) = 0;


	/**
	 * *[Analysis mode]* Computes the generative process off-line
	 * */
	virtual void a_predict() = 0;

	/**
	 * *[Analysis mode]* Save off-line computations to disk
	 * @param path Directory full path
	 * */
	virtual void a_save(string path) = 0;

	// ------------------------- Experiment mode methods -------------------------


	/**
	 * *[Experiment mode]* Enables the experiment mode
	 * @param pID Primitive ID
	 * @param winSize Sliding window size
	 * @param param Input container with parameters
	 * @param nT Number of experiment times
	 * @param store_gen A flag indicating to store the generative process states. This should be used with caution,
	 *     since memory allocation for large data may degrade performance. It is recommended to use @ref e_getState
	 *     to get the current state of the network, without storing information on the back-end side.
	 *     Hence, the client front-end does the storage.
	 * @param store_inf A flag indicating to store the inference process states. Valuable only if the parameter *store_gen*
	 *     is set true. Thus should be used only for debugging, since memory allocation for large data may degrade performance.
	 *     All the computation steps for the *back propagation trough time* (BPPT) algorithm are stored.
	 * */
	virtual void e_enable(int pID, int winSize, float param, int nT, bool store_gen, bool store_inf) = 0;

	/**
	 * *[Experiment mode]* Computes one prediction with the generative process
	 * */
	virtual void e_generate() = 0;

	/**
	 * *[Experiment mode]* Initialize the inference process forward computations
	 * */
	virtual void e_initForward() = 0;

	/**
	 * *[Experiment mode]* Computes forward inference process
	 * */
	virtual void e_forward() = 0;

	/**
	 * *[Experiment mode]* Initializes the backward inference process computation
	 * */
	virtual void e_initBackward() = 0;

	/**
	 * *[Experiment mode]* Computes the backward pass of the BPTT algorithm for performing inference
	 * @param time Current time step
	 * */
	virtual void e_backward(int time) = 0;
	/**
	 * *[Experiment model]* Copies the parameters with optimal loss function (greedy optimization)
	 * */
	virtual void e_copyParam() = 0;

	/**
	 * *[Experiment model]* Persists the parameters with optimal loss function (greedy optimization)
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
	 * *[Experiment mode]* Writes the current layer state to a float array.
	 * Both the generation and inference latent states are provided real time for analysis.
	 * @param output Float array reference. The size of the array can be obtained from @ref getStateDim
	 * @return Pointer to the next writable position in the float array
	 * */
	virtual float* e_getState(float* output) = 0;

	/**
	 * *[Experiment mode]* Save the network model
	 * @param path Output directory full path
	 * */
	virtual void e_save(string path) = 0;

	// ------------------------- debug methods -------------------------

	/**
	 * Shows the network state in the standard output
	 * */
	virtual void print() = 0;

	/**
	 * Virtual destructor
	 * */
	virtual ~ILayer(){}
};

} /* namespace oist */

#endif /* NETWORK_LAYER_ILAYER_H_ */
