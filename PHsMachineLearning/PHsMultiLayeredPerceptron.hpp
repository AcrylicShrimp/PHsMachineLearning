
/*
	2016.09.15
	Created by PHJ.
*/

#ifndef _PHS_MULTI_LAYERED_PERCEPTRON_HPP

#define _PHS_MULTI_LAYERED_PERCEPTRON_HPP

namespace PHs
{
	/*
		TODO : Place your code here
	*/
	
	inline namespace PHsMachineLearning
	{
		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::MultiLayeredPerceptron() :
			nLearningRate{1.0f}
		{
			std::mt19937 sEngine;
			std::uniform_real_distribution <float> sGenerator{-0.5f, 0.5f};

			for(auto &sWeightArray : this->sFirstWeightArray)
				for(auto &nWeight : sWeightArray)
					nWeight = sGenerator(sEngine);
					//nWeight = 0.0f;

			for(auto &sWeightArray : this->sSecondWeightArray)
				for(auto &nWeight : sWeightArray)
					nWeight = sGenerator(sEngine);
					//nWeight = 0.0f;
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::MultiLayeredPerceptron(MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &&sNewMultiLayeredPerceptron) :
			nLearningRate{sNewMultiLayeredPerceptron.nLearningRate},
			sInputList{std::move(sNewMultiLayeredPerceptron.sInputList)},
			sFirstWeightArray{std::move(sNewMultiLayeredPerceptron.sFirstWeightArray)},
			sSecondWeightArray{std::move(sNewMultiLayeredPerceptron.sSecondWeightArray)}
		{
			//Empty.
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::MultiLayeredPerceptron(const MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &sNewMultiLayeredPerceptron) :
			nLearningRate{sNewMultiLayeredPerceptron.nLearningRate},
			sInputList{sNewMultiLayeredPerceptron.sInputList},
			sFirstWeightArray{sNewMultiLayeredPerceptron.sFirstWeightArray},
			sSecondWeightArray{sNewMultiLayeredPerceptron.sSecondWeightArray}
		{
			//Empty.
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::operator=(MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &&sNewMultiLayeredPerceptron)
		{
			this->nLearningRate = sNewMultiLayeredPerceptron.nLearningRate;
			this->sInputList = std::move(sNewMultiLayeredPerceptron.sInputList);
			this->sFirstWeightArray = std::move(sNewMultiLayeredPerceptron.sFirstWeightArray);
			this->sSecondWeightArray = std::move(sNewMultiLayeredPerceptron.sSecondWeightArray);
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::operator=(const MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength> &sNewMultiLayeredPerceptron)
		{
			this->nLearningRate = sNewMultiLayeredPerceptron.nLearningRate;
			this->sInputList = sNewMultiLayeredPerceptron.sInputList;
			this->sFirstWeightArray = sNewMultiLayeredPerceptron.sFirstWeightArray;
			this->sSecondWeightArray = sNewMultiLayeredPerceptron.sSecondWeightArray;
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		void MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::addCase(const Array <InputLength> &sInputVector, const Array <OutputLength> &sDesiredOutputVector)
		{
			this->sInputList.emplace_back(sInputVector, sDesiredOutputVector);
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		void MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::startLearn(float nThreshold)
		{
			bool bDone;

			do
			{
				bDone = true;

				for(const auto &sInputPair : this->sInputList)
				{
					std::array <float, OutputLength> sOutput;
					this->calcOutput(sInputPair.first, sOutput);

					for(auto nOutputIndex = 0u ; nOutputIndex < OutputLength ; ++nOutputIndex)
					{
						bDone &= std::abs(sInputPair.second[nOutputIndex] - sOutput[nOutputIndex]) <= nThreshold;

						float nDelta = (sInputPair.second[nOutputIndex] - sOutput[nOutputIndex]) * (1.0f - sOutput[nOutputIndex]) * sOutput[nOutputIndex];

						for(auto nHiddenIndex = 0u ; nHiddenIndex < HiddenLength ; ++nHiddenIndex)
							this->sSecondWeightArray[nHiddenIndex + 1u][nOutputIndex] +=
								this->nLearningRate *
								this->calcOutputFromHidden(sInputPair.first, nHiddenIndex) *
								nDelta;

						this->sSecondWeightArray[0u][nOutputIndex] += this->nLearningRate * nDelta;
					}

					for(auto nHiddenIndex = 0u ; nHiddenIndex < HiddenLength ; ++nHiddenIndex)
					{
						float nDelta{0.0f};

						for(auto nOutputIndex = 0u ; nOutputIndex < OutputLength ; ++nOutputIndex)
							nDelta += (sInputPair.second[nOutputIndex] - sOutput[nOutputIndex]) *
								(1.0f - sOutput[nOutputIndex]) *
								sOutput[nOutputIndex] *
								this->sSecondWeightArray[nHiddenIndex + 1u][nOutputIndex];

						{
							float nOutput = this->calcOutputFromHidden(sInputPair.first, nHiddenIndex);
							nDelta *= nOutput * (1.0f - nOutput);
						}

						for(auto nInputIndex = 0u ; nInputIndex < InputLength ; ++nInputIndex)
							this->sFirstWeightArray[nInputIndex + 1u][nHiddenIndex] += this->nLearningRate * sInputPair.first[nInputIndex] * nDelta;

						this->sFirstWeightArray[0u][nHiddenIndex] += this->nLearningRate * nDelta;
					}
				}
			}
			while(!bDone);
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		void MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::calcOutput(const Array <InputLength> &sInputVector, Array <OutputLength> &sOutputVector) const
		{
			for(auto nIndex = 0u ; nIndex < OutputLength ; ++nIndex)
				sOutputVector[nIndex] = this->calcOutputFromOutput(sInputVector, nIndex);
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		float MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::calcOutputFromHidden(const Array <InputLength> &sInputVector, std::uint32_t nIndex) const
		{
			assert(nIndex < HiddenLength);

			float nOutput{0.0f};

			for(auto nInputIndex = 0u ; nInputIndex < InputLength ; ++nInputIndex)
				nOutput += sInputVector[nInputIndex] * this->sFirstWeightArray[nInputIndex + 1u][nIndex];

			nOutput += this->sFirstWeightArray[0][nIndex];

			return 1.0f / (1.0f + std::exp(-nOutput));
		}

		template <std::uint32_t InputLength, std::uint32_t HiddenLength, std::uint32_t OutputLength>
		float MultiLayeredPerceptron <InputLength, HiddenLength, OutputLength>::calcOutputFromOutput(const Array <InputLength> &sInputVector, std::uint32_t nIndex) const
		{
			assert(nIndex < OutputLength);

			float nOutput{0.0f};

			for(auto nHiddenIndex = 0u ; nHiddenIndex < HiddenLength ; ++nHiddenIndex)
				nOutput += this->calcOutputFromHidden(sInputVector, nHiddenIndex) * this->sSecondWeightArray[nHiddenIndex + 1u][nIndex];

			nOutput += this->sSecondWeightArray[0][nIndex];

			return 1.0f / (1.0f + std::exp(-nOutput));
		}
	}
}

#endif