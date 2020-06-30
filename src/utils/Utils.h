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

#ifndef SRC_UTILS_UTILS_H_
#define SRC_UTILS_UTILS_H_

#include "../includes.h"

namespace oist {

/**
 * This class provides functionality for string processing, data saving and loading.
 * It implements a singleton design pattern
 * */
class Utils {

	//Singleton instance
	static Utils* myInstance;
	const string delimiter;
	Utils();
	~Utils();
	static std::default_random_engine generator;
	static std::normal_distribution<float> distribution;

public:

	enum nonlinearity {Linear=0, Conv, Sigmoid, Tanh, ReLU, ReLU_Leaky};

	/**
	 * Gets the singleton instance
	 * */
	static Utils* getInstance();

	/**
	 * Trim chars from the left
	 * @param io Input/output string
	 * @param chars Chars to be removed
	 * */
	void ltrim(string& io, const string& chars);

	/**
	 * Trim chars from the right
	 * @param io Input/output string
	 * @param chars Chars to be removed
	 * */
	void rtrim(string& io, const string& chars);

	/**
	 * Trim chars from both sides
	 * @param io Input/output string
	 * @param chars Chars to be removed
	 * */
	void trim(string& io, const string& chars);

	/**
	 * Low caps a string
	 * @param input Input string
	 * */
	void tolower(string& input);

	/**
	 * Splits a string and feed the tokens to a vector
	 * @param output Output vector
	 * @param input Input text string
	 * */
	void split(vector<string>& output, string input);

	/**
	 * Gets the file delimiter string to read/write data
	 * */
	string getDelimiter();
	/**
	 * kaiming_uniform initialization for weight matrices
	 * @param iDim Input space dimension
	 * @param oDim Output space dimension
	 * @param type Type of non-linearity
	 * */
	MatrixXf kaiming_uniform_initialization(int iDim, int oDim, nonlinearity type);

	/**
	 * kaiming_uniform initialization for biases
	 * @param dim Bias vector dimension
	 * */
	VectorXf kaiming_uniform_initialization(int dim);

	/**
	 * Loads the property file
	 * @param mapFloat1D Output container for float1D map parameters
	 * @param mapString Output container for string map parameters
	 * @param mapBool Output container for bool map parameters
	 * @param path Full path to the properties
	 * */
	void getProperties(map<string,float1DContainer>& mapFloat1D, map<string,string>& mapString, map<string,bool>& mapBool, string path);

	/**
	 * Computes the power operation
	 * @param io Input element
	 * @param exp Power exponent
	 * */
	template <typename T> void power(T* io, float exp);

	/**
	 * Computes the softmax function
	 * @param io Input/Output data type
	 * */
	template <typename T> void softmax(T* io);

	/**
	 * Computes Gaussian noise in N(1,0)
	 * @param io Input/Output data type
	 * */
	template <typename T> void randN(T* io);

	/**
	 * Shuffles a container
	 * @param io Input/Output data type
	 * */
	template <typename T> void shuffle(T* io);

	/**
	 * Sets the element to zero
	 * @param io Input/Output data type
	 * */
	template <typename T> void zero(T* io);

	/**
	 * Computes the hyperbolic tangent
	 * @param io Input/Output data type
	 * */
	template <typename T> void tanH(T* io);

	/**
	 * Computes ADAM optimization
	 * @param p Input/Ouput parameter
	 * @param g Input gradient
	 * @param m Input m
	 * @param v Input v
	 * @param epoch Epoch number
	 * @param alpha Adam optimization hyper parameter \f$\alpha\f$
	 * @param beta1 Adam optimization hyper parameter \f$\beta_1\f$
	 * @param beta2 Adam optimization hyper parameter \f$\beta_2\f$
	 * */
	template <typename T> void adam(T* p, T* g, T* m, T* v, int epoch, float alpha, float beta1, float beta2);

	/**
	 * Copy values from two Eigen objects
	 * @param input Input data
	 * @param output Output data
	 * */
	template <typename T> void copyEigen(T* input, T* output);

	/**
	 * Copies Eigen data to a float array
	 * @param eInput Eigen data pointer
	 * @param fOutput Float array pointer
	 * @return pointer to the buffer next position
	 * */
	template <typename T> float* copyEigenData(T* eInput, float* fOutput);


	/**
	 * Saves Eigen data to file
	 * @param file File object pointer
	 * @param eInput Eigen data
	 * @param delimiter String delimiter
	 * */
	template <typename T> void saveEigen(ofstream* file, T* eInput, string delimiter);

	/**
	 * Saves scalar data to file
	 * @param file File object  pointer
	 * @param input Scalar to be saved
	 * */
	template <typename T> void saveScalar(ofstream* file, T input);

	/**
	 * Saves a data container to file
	 * @param file File object pointer
	 * @param input Pointer to data collection
	 * @param delimiter String delimiter
	 * */
	template <typename T> void saveContainer(ofstream* file, T* input, string delimiter);

	/**
	 * Loads Eigen data to from file
	 * @param file File object pointer
	 * @param output Pointer to data collection
	 * @param delimiter String delimiter
	 * */
	template <typename T> void loadEigen(ifstream* file, T* output, string delimiter);

	/**
	 * Loads container data from file
	 * @param file File object pointer
	 * @param output Pointer to data collection
	 * @param delimiter String delimiter
	 * */
	template <typename T> void loadContainer(ifstream* file, T* output, string delimiter);

	/**
	 * Loads a scalar from file
	 * @param file File object pointer
	 * @param output Pointer to the scalar
	 * */
	template <typename T> void loadScalar(ifstream* file, T* output);

};

	// #################################################################################################
	// NOTE: For shared library compilation the implementation of the template functions are provided
	// in the header file. In case of stand-alone compilation, this code should be moved to Utils.cpp
	// #################################################################################################

	template <typename T>
	inline void Utils::power(T* _v, float _p){
		auto d = _v->data();
		for (int i = 0; i < _v->size(); i++, d++)
			*d = pow(*d,_p);
	}

	template <typename T>
	inline void Utils::softmax(T* _v){
		auto d = _v->data();
		float accum = 0.0;
		for (int i = 0; i < _v->size(); i++, d++){
			*d = exp(*d);
			accum += *d;
		}
		*_v /= accum;
	}

	template <typename T>
	inline void Utils::randN(T* _v){
		auto d = _v->data();
		for (int i = 0; i < _v->size(); i++, d++)
			*d = distribution(generator);
	}

	template <typename T>
	inline void Utils::shuffle(T* _v){
		std::shuffle(std::begin(*_v), std::end(*_v), generator);
	}

	template <typename T>
	inline void Utils::zero(T* _v){
		auto d = _v->data();
		for (int i = 0; i < _v->size(); i++,d++){
			*d = 0.0f;
		}
	}

	template <typename T>
	inline void Utils::tanH(T* _v){
		auto d = _v->data();
		for (int i = 0; i < _v->size(); i++, d++)
			*d = tanh(*d);
	}

	template <typename T>
	inline void Utils::adam(T* _p, T* _g, T* _m, T* _v, int _epoch, float _alpha, float _beta1, float _beta2){
		auto p = _p->data();
		auto g = _g->data();
		auto m = _m->data();
		auto v = _v->data();
		for (int i = 0 ; i < _p->size(); i++, p++, g++, m++, v++){
			*m = _beta1*(*m) + (1.0-_beta1)*(*g);
			*v = _beta2*(*v) + (1.0-_beta2)*((*g) * (*g));
			float mHat = *m / (1.0 - pow(_beta1, _epoch));
			float vHat = *v / (1.0 - pow(_beta2, _epoch));
			*p -= _alpha *mHat/(pow(vHat, 0.5)+NON_ZERO);
		}
	 }

	template <typename T>
	inline void Utils::copyEigen(T* _from, T* _to){
		auto f = _from->data();
		auto t = _to->data();
		for (int i = 0; i < _from->size(); i++, f++, t++)
			*t = *f;
	}

	template <typename T>
	void Utils::saveEigen(ofstream* _f, T* _d, string _delimiter){
		auto d = _d->data();
		for (int j = 0; j < _d->size()-1; j++, d++){
			*_f << *d << _delimiter;
		}
		*_f << *d << endl;
	}

	template <typename T>
	float* Utils::copyEigenData(T* _e, float* _f){
		auto d = _e->data();
		for (unsigned int j = 0; j < _e->size(); j++, d++, _f++){
			*_f = *d;
		}
		return _f;
	}

	template <typename T>
	void Utils::saveScalar(ofstream* _f, T _v){
		if (std::is_same<T, bool>::value)
			*_f << (_v ? "true":"false") << endl;
		else
			*_f << _v << endl;
	}

	template <typename T>
	void Utils::saveContainer(ofstream* _f, T* _v, string _delimiter){
		auto d = _v->begin();
		for (unsigned int j = 0; j < _v->size()-1; j++, d++){
			*_f << *d << _delimiter;
		}
		*_f << *d << endl;
	}

	template <typename T>
	void Utils::loadEigen(ifstream* _f, T* _v, string _delimiter){
		try{
			string line;
			getline(*_f, line);
			size_t pos = 0;
			std::string::size_type sz;
			std::string token;
			auto d = _v->data();
			while ((pos = line.find(_delimiter)) != std::string::npos) {
				 token = line.substr(0, pos);
				 try{
					 *d = std::stof(token,&sz);
					 d++;
				 }
				 catch (const std::invalid_argument& ia) {
					  std::cerr << "Invalid argument: " << ia.what() << '\n';
				 }catch (const std::out_of_range& oor) {
					 *d = 0.0;
					 d++;
				 }
				 line.erase(0, pos + _delimiter.length());
			 }
			try{
				 *d = std::stof(line,&sz);
			}
			catch (const std::invalid_argument& ia) {
				  std::cerr << "Invalid argument: " << ia.what() << '\n';
			}catch (const std::out_of_range& oor) {
				 *d = 0.0;
			}
		}catch(...){
			throw Exception("A exception occurred in readVecFromFile method");
		}
	}

	template <typename T>
	void Utils::loadContainer(ifstream* _f, T* _v, string _delimiter){
		try{
			_v->clear();
			string line;
			getline(*_f, line);
			size_t pos = 0;
			std::string::size_type sz;
			std::string token;

			while ((pos = line.find(_delimiter)) != std::string::npos) {
				 token = line.substr(0, pos);
				 try{
					 _v->push_back(std::stof(token,&sz));
				 }
				 catch (const std::invalid_argument& ia) {
					  std::cerr << "Invalid argument: " << ia.what() << '\n';
				 }catch (const std::out_of_range& oor) {
					  std::cerr << "Out of Range error: " << oor.what() << " token ["<< token <<  "]" << endl;
				 }
				 line.erase(0, pos + _delimiter.length());
			 }
			try{
				_v->push_back(std::stof(line,&sz));
			}
			catch (const std::invalid_argument& ia) {
				  std::cerr << "Invalid argument: " << ia.what() << '\n';
			}catch (const std::out_of_range& oor) {
			  std::cerr << "Out of Range error: " << oor.what() << " token ["<< token <<  "]" << endl;
			}
		}catch(...){
			throw Exception("A exception occurred in readVecFromFile method");
		}
	}

	template <typename T>
	void Utils::loadScalar(ifstream* _f, T* _v){
		string line;
		getline(*_f, line);
		std::string::size_type sz;
			try{
				if (std::is_same<T, int>::value)
					*_v = std::stoi(line,&sz);
				if (std::is_same<T, float>::value)
					*_v = std::stof(line,&sz);
				if (std::is_same<T, bool>::value)
					*_v = (line == "true");
			}
			 catch (const std::invalid_argument& ia) {
				  std::cerr << "Invalid argument: " << ia.what() << '\n';
			 }catch (const std::out_of_range& oor) {
				  std::cerr << "Out of Range error: " << oor.what() << " token ["<< line <<  "]" << endl;
			 }
		} 
	



} /* namespace oist */

#endif /* SRC_UTILS_UTILS_H_ */
