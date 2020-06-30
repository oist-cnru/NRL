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

#ifndef SRC_STAND_UTILS_H_
#define SRC_STAND_UTILS_H_

namespace oist {

	/**
	 * This class implements a circular buffer (queue)
	 * */
	class CircularBuffer{

		int dataSize;
		int maxSize;
		int csize;
		int nDof;
		float* data;

	public: 

	/**
	 * Constructor
	 * @param size Maximum buffer size
	 * @param nDOF Number of degrees of freedom (the dimension of data to be queued)
	 * */
	CircularBuffer(int size, int nDOF);

	/**
	 * Push data at the end of the buffer.
	 * Automatically dequeues if the buffer is full.
	 * @param input Input data array of \f$dimension=size nDOF\f$
	 * */
	void push(float* input);

	/**
	 * Gets the current size of the buffer
	 * */
	int size();

	/**
	 * Gets a pointer to the first element of data stored
	 * @return Pointer to the first element on the data array
	 * */
	float* getData();


	/**
	 * Destructor
	 * */
	~CircularBuffer();

	};


	CircularBuffer::CircularBuffer(int _maxSize, int _nDof){

		maxSize = _maxSize;
		nDof = _nDof;
		dataSize = maxSize * nDof;
		csize = 0;
		data = new float[dataSize];
		for (int i = 0; i < dataSize; i++)
			data[i] = 0.0;

	}

	void CircularBuffer::push(float* _input){

		csize += 1;

		float* w_i1 = &data[0];

		// updating the window queue 
		if (csize >= maxSize){

			csize = maxSize;
			float* w_i2 = w_i1 + nDof;

			// dequeuing the last element
			for (int i = nDof ; i < dataSize; i++, w_i1++, w_i2++){
				*w_i1 = *w_i2;
			}		
		}else{
			w_i1 += csize*nDof;

		}
	
		// queuing the newly generated target
		float* w_i2 = &_input[0];				
		for (int i = 0 ; i < nDof; i++, w_i1++, w_i2++){
			*w_i1 = *w_i2;
		}
	}

	int CircularBuffer::size(){

		return csize;
	}

	float* CircularBuffer::getData(){

		return data;
	}

	CircularBuffer::~CircularBuffer(){

		delete[] data;
	}

} /* namespace oist */

#endif /* SRC_STAND_UTILS_H_ */
