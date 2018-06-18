
/*
	2016.09.15
	Created by PHJ.
*/

#ifndef _PHS_MULTI_LAYERED_PERCEPTRON_H

#define _PHS_MULTI_LAYERED_PERCEPTRON_H

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <utility>
#include <vector>

namespace PHs
{
	/*
		TODO : Place your code here
	*/

	inline namespace PHsMachineLearning
	{
		template <std::uint32_t Length> using Array = std::array <float, Length>;

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength> class MultiLayeredPerceptron
		{
		private:
			float nLearningRate;
			std::vector <std::pair <Array <InputLength>, Array <OutputLength>>> sInputList;
			std::array <std::array <float, HiddenLength>, InputLength + 1u> sFirstWeightArray;
			std::array <std::array <float, OutputLength>, HiddenLength + 1u> sSecondWeightArray;

		public:
			MultiLayeredPerceptron();
			MultiLayeredPerceptron(MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &&sNewMultiLayeredPerceptron);
			MultiLayeredPerceptron(const MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &sNewMultiLayeredPerceptron);
			~MultiLayeredPerceptron() = default;

		public:
			MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &operator=(MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &&sNewMultiLayeredPerceptron);
			MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &operator=(const MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &sNewMultiLayeredPerceptron);

			void addCase(const Array <InputLength> &sInputVector, const Array <OutputLength> &sDesiredOutputVector);
			void startLearn(float nThreshold);
			void calcOutput(const Array <InputLength> &sInputVector, Array <OutputLength> &sOutputVector) const;

			inline float &learning_rate();
			inline float learning_rate() const;
			
		private:
			float calcOutputFromHidden(const Array <InputLength> &sInputVector, std::uint32_t nIndex) const;
			float calcOutputFromOutput(const Array <InputLength> &sInputVector, std::uint32_t nIndex) const;
		};

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		inline float &MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::learning_rate()
		{
			return this->nLearningRate;
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		inline float MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::learning_rate() const
		{
			return this->nLearningRate;
		}
	}
}

#include "PHsMultiLayeredPerceptron.hpp"

#endif