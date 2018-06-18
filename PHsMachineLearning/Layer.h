
/*
	2017.02.01
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_PHS_MACHINELEARNING_LAYER_H

#define _CLASS_PHS_MACHINELEARNING_LAYER_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace PHs::MachineLearning
{
	class Layer
	{
	private:
		std::vector<std::vector<float>> sMatrix;
		std::vector<float> sBias;
		std::vector<std::vector<float>> sErrorMatrix;
		std::vector<float> sOutput;
		Layer *pPrevLayer;
		Layer *pNextLayer;

	public:
		Layer();
		Layer(uint32_t nNodeCount);
		Layer(const Layer &sSrc);
		Layer(Layer &&sSrc);
		~Layer() = default;

	public:
		Layer &operator=(const Layer &sSrc);
		Layer &operator=(Layer &&sSrc);

	public:
		void linkLayer(Layer &sNextLayer);
		void linkLayerWithoutModify(Layer &sNextLayer);
		void resetLayer(std::mt19937_64 &sEngine, std::normal_distribution<float> &sDist);
		void resetLayer(std::mt19937_64 &sEngine, std::uniform_real_distribution <float> &sDist);
		void resetLayerAll(std::mt19937_64 &sEngine, std::normal_distribution<float> &sDist);
		void resetLayerAll(std::mt19937_64 &sEngine, std::uniform_real_distribution<float> &sDist);
		void resetLayerAll(float nSigma);
		void resetLayerAll(float nMin, float nMax);
		void calcOutput();
		void calcOutputAll();
		void calcOutputAll(const std::vector<float> &sInputVector);
		float calcError(const std::vector<float> &sDesiredOutputVector);
		void clearError();
		void accrueError(const std::vector<float> &sDesiredOutputVector);
		void learnToBack(float nLearningRate, float nLambda, float nWeightSum, uint32_t nInputListCount);
		void learnToBackAll(float nLearningRate, float nLambda, float nWeightSum, uint32_t nInputListCount);
		void learnToBackAll(float nLearningRate, float nLambda, uint32_t nInputListCount);
		static void learnToBackAll(Layer &sInputLayer, Layer &sOutputLayer, float nLearningRate, float nLambda, uint32_t nEpochCount, const std::vector<std::vector<float>> &sInputVectorList, const std::vector<std::vector<float>> &sDesiredOutputVectorList);

		inline Layer *getPrevLayer() const;
		inline Layer *getNextLayer() const;
		inline uint32_t getNodeCount() const;
		inline std::vector<std::vector<float>> &getMatrix();
		inline const std::vector<std::vector<float>> &getMatrix() const;
		inline std::vector<float> &getBias();
		inline const std::vector<float> &getBias() const;
		inline std::vector<float> &getOutputVector();
		inline const std::vector<float> &getOutputVector() const;
	};

	inline Layer *Layer::getPrevLayer() const
	{
		return this->pPrevLayer;
	}

	inline Layer *Layer::getNextLayer() const
	{
		return this->pNextLayer;
	}

	inline uint32_t Layer::getNodeCount() const
	{
		return this->sMatrix.size();
	}

	inline std::vector<std::vector<float>> &Layer::getMatrix()
	{
		return this->sMatrix;
	}

	inline const std::vector<std::vector<float>> &Layer::getMatrix() const
	{
		return this->sMatrix;
	}

	inline std::vector<float> &Layer::getBias()
	{
		return this->sBias;
	}

	inline const std::vector<float> &Layer::getBias() const
	{
		return this->sBias;
	}

	inline std::vector<float> &Layer::getOutputVector()
	{
		return this->sOutput;
	}

	inline const std::vector<float> &Layer::getOutputVector() const
	{
		return this->sOutput;
	}
}

#endif