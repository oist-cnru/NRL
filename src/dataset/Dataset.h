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

#ifndef SRC_DATASET_DATASET_H_
#define SRC_DATASET_DATASET_H_

#include "../includes.h"
#include "../utils/Utils.h"
#include "../robot/IRobot.h"

namespace oist {

/**
 * This class provides the functionalities for encoding and decoding robot and network data
 * */
class Dataset {

private:

	Utils* ut;
	string path;
	string dataPrefix;

	// ---- robot Joints
	float1DContainer jmin;
	float1DContainer jmax;
	float1DContainer jrange;
	float2DContainer ref;
	int nDof;
	int1DContainer nUnits; // number of encoding units per degree of freedom in the softmax neuron space

	// --- softmax neuron encoding
	float dsoft;    		// distance between references in the joint space (1 neuron each dsoft units)
	float sigma;	  		// variance in the neuron space
	int encDim;		  	// number of neuron units for all DOFs
	int nPrims; 	  		// Number of primitives
	int seqLen;  			// Length of sequences
	vector<int> nSamples;  	// Number of samples per primitive;

	void loadData(string _path, string _dataPrefix, int _nPrim, int1DContainer _nSamples, float4DContainer& _data);
	void softmax(float2DContainer& _data, vectorXf2DContainer& _encData);

public:

	/**
	 * Constructor
	 * @param path Full path to the dataset
	 * @param prefix Primitive filename prefix
	 * @param samples Integer container with number of sample per primitives
	 * @param dsoft The desired encoding range in the joint space for a neuron.
	 *     That is, for a revolute joint, if *dsoft=10* one neuron unit would encode 10 degrees.
	 * @param sigma Parameter \f$\sigma\f$ for the activation of the \f$i^\mathrm{th}\f$ output unit,
	 *     related to the \f$j^\mathrm{th}\f$ dimension of the joint space, such that
	 *     \f[o_{j,i} =  \mathrm{exp}\left(\frac{-(x_j- \bar{x_i})^2}{\sigma^2}\right)\frac{1}{Z_j}\f]
	 *     where \f$x_j\f$ is the observation in the joint space, \f$\bar{x_i}\f$ is the reference value,
	 *     and the softmax normalization term is
	 *     \f[Z_j= \sum_i \mathrm{exp}\left(\frac{-(x_j- \bar{x_i})^2}{\sigma^2}\right)\f]
	 * @param ut Pointer to a Utils instance
	 * @param robot Pointer to a IRobot extended instance
	 * */
	Dataset(string path, string prefix, int1DContainer& samples, float dsoft, float sigma, Utils* ut, IRobot* robot);

	/**
	 * Computes softmax encoding
	 * @param output Container to store encoded data
	 * */
	void encodeSoftmax(vectorXf4DContainer& output);

	/**
	 * Computes softmax encoding
	 * @param input Pointer to the 1D input array data
	 * @param size Pointer to an array containing the number time steps and the
	 *     number of degrees of freedom in the input data
	 * @param output Container to store encoded data
	 * */
	void encodeSoftmax(float* input, int* size, vectorXf2DContainer& output);

	/**
	 * Transforms back from softmax encoding of dimension *j* to a scalar value for the robot joint *j*
	 * @param input Input array with encoded data
	 * @param dim Dimension of the joint space
	 * @return Decoded value in the joint space
	 * */
	float decodeSoftmax(ArrayXf& input, int dim);

	/**
	 * Gets the number of neurons per dimension
	 * @param output Output dimension container
	 * */
	void getNunitsPerDim(int1DContainer& output);

	/**
	 * Gets the length of the sequences in the data-set, assumed to be of same length
	 * @return primitive length
	 * */
	int getPrimLength();

	/**
	 * Gets the number of primitives in the data-set
	 * @return number of primitives
	 * */
	int getNPrim();

	virtual ~Dataset();

};

	// #################################################################################################
	// NOTE: For shared library compilation the implementation of inline methods are provided
	// in the header file. In case of stand-alone compilation, this code should be moved to Utils.cpp
	// #################################################################################################
	inline float Dataset::decodeSoftmax(ArrayXf& input, int dim){

		float1DContainer ref_ = ref[dim];

		auto d = input.data();
		float1DContainer::iterator r =ref_.begin();

		float dec = 0.0;

		for (; r != ref_.end(); r++, d++){
			dec += (*r)*(*d);
		}
		dec *= jrange[dim];
		dec += jmin[dim];
		return dec;

	}


} /* namespace oist */

#endif /* SRC_DATASET_DATASET_H_ */

