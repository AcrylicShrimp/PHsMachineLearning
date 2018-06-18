
/*
	2017.02.01
	Created by AcrylicShrimp.
*/

#include "Layer.h"

namespace PHs::MachineLearning
{
	Layer::Layer() :
		pPrevLayer{nullptr},
		pNextLayer{nullptr}
	{
		//Empty.
	}

	Layer::Layer(uint32_t nNodeCount) :
		sMatrix(nNodeCount),
		sOutput(nNodeCount),
		pPrevLayer{nullptr},
		pNextLayer{nullptr}
	{
		//Empty.
	}

	Layer::Layer(const Layer &sSrc) :
		sMatrix{sSrc.sMatrix},
		sBias{sSrc.sBias},
		sErrorMatrix{sSrc.sErrorMatrix},
		sOutput{sSrc.sOutput},
		pPrevLayer{sSrc.pPrevLayer},
		pNextLayer{sSrc.pNextLayer}
	{
		//Empty.
	}

	Layer::Layer(Layer &&sSrc) :
		sMatrix{std::move(sSrc.sMatrix)},
		sBias{std::move(sSrc.sBias)},
		sErrorMatrix{std::move(sSrc.sErrorMatrix)},
		sOutput{std::move(sSrc.sOutput)},
		pPrevLayer{sSrc.pPrevLayer},
		pNextLayer{sSrc.pNextLayer}
	{
		//Empty.
	}

	Layer &Layer::operator=(const Layer &sSrc)
	{
		this->sMatrix = sSrc.sMatrix;
		this->sBias = sSrc.sBias;
		this->sErrorMatrix = sSrc.sErrorMatrix;
		this->sOutput = sSrc.sOutput;
		this->pPrevLayer = sSrc.pPrevLayer;
		this->pNextLayer = sSrc.pNextLayer;

		return *this;
	}

	Layer &Layer::operator=(Layer &&sSrc)
	{
		this->sMatrix = std::move(sSrc.sMatrix);
		this->sBias = std::move(sSrc.sBias);
		this->sErrorMatrix = std::move(sSrc.sErrorMatrix);
		this->sOutput = std::move(sSrc.sOutput);
		this->pPrevLayer = sSrc.pPrevLayer;
		this->pNextLayer = sSrc.pNextLayer;

		return *this;
	}

	void Layer::linkLayer(Layer &sNextLayer)
	{
		this->pNextLayer = &sNextLayer;
		sNextLayer.pPrevLayer = this;

		uint32_t nNextNodeCount = sNextLayer.getNodeCount();

		for(auto &sFactor : this->sMatrix)
			sFactor.resize(nNextNodeCount);

		this->sBias.resize(nNextNodeCount, .0f);

		uint32_t nNodeCount = this->getNodeCount();

		if(this->pPrevLayer)
		{
			this->sErrorMatrix.resize(1u);
			this->sErrorMatrix.back().resize(nNodeCount);
		}

		sNextLayer.sErrorMatrix.resize(nNodeCount + 1u);
		for(auto &sError : sNextLayer.sErrorMatrix)
			sError.resize(nNextNodeCount);
	}

	void Layer::linkLayerWithoutModify(Layer &sNextLayer)
	{
		this->pNextLayer = &sNextLayer;
		sNextLayer.pPrevLayer = this;

		uint32_t nNodeCount = this->getNodeCount();

		if(this->pPrevLayer)
		{
			this->sErrorMatrix.resize(1u);
			this->sErrorMatrix.back().resize(nNodeCount);
		}

		sNextLayer.sErrorMatrix.resize(nNodeCount + 1u);
		for(auto &sError : sNextLayer.sErrorMatrix)
			sError.resize(sNextLayer.getNodeCount());
	}

	void Layer::resetLayer(std::mt19937_64 &sEngine, std::normal_distribution<float> &sDist)
	{
		for(auto &nFactor : this->sBias)
			nFactor = sDist(sEngine);

		for(auto &sFactor : this->sMatrix)
			for(auto &nFactor : sFactor)
				nFactor = sDist(sEngine);
	}

	void Layer::resetLayer(std::mt19937_64 &sEngine, std::uniform_real_distribution <float> &sDist)
	{
		for(auto &nFactor : this->sBias)
			nFactor = sDist(sEngine);

		for(auto &sFactor : this->sMatrix)
			for(auto &nFactor : sFactor)
				nFactor = sDist(sEngine);
	}

	void Layer::resetLayerAll(std::mt19937_64 &sEngine, std::normal_distribution<float> &sDist)
	{
		this->resetLayer(sEngine, sDist);

		if(this->pNextLayer)
			this->pNextLayer->resetLayerAll(sEngine, sDist);
	}

	void Layer::resetLayerAll(std::mt19937_64 &sEngine, std::uniform_real_distribution<float> &sDist)
	{
		this->resetLayer(sEngine, sDist);

		if(this->pNextLayer)
			this->pNextLayer->resetLayerAll(sEngine, sDist);
	}

	void Layer::resetLayerAll(float nSigma)
	{
		std::mt19937_64 sEngine{static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count())};
		std::normal_distribution<float> sDist{0.f, nSigma};

		this->resetLayer(sEngine, sDist);

		if(this->pNextLayer)
			this->pNextLayer->resetLayerAll(sEngine, sDist);
	}

	void Layer::resetLayerAll(float nMin, float nMax)
	{
		std::mt19937_64 sEngine{static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count())};
		std::uniform_real_distribution<float> sDist{nMin, nMax};

		this->resetLayer(sEngine, sDist);

		if(this->pNextLayer)
			this->pNextLayer->resetLayerAll(sEngine, sDist);
	}

	void Layer::calcOutput()
	{
		if(!this->pNextLayer)
			return;

		for(auto &nOutput : this->pNextLayer->sOutput)
			nOutput = .0f;

		for(uint32_t nOutputIndex = 0u, nOutputCount = this->pNextLayer->sMatrix.size() ; nOutputIndex < nOutputCount ; ++nOutputIndex)
		{
			for(uint32_t nInputIndex = 0u, nInputCount = this->sMatrix.size() ; nInputIndex < nInputCount ; ++nInputIndex)
				this->pNextLayer->sOutput[nOutputIndex] += this->sMatrix[nInputIndex][nOutputIndex] * this->sOutput[nInputIndex];

			this->pNextLayer->sOutput[nOutputIndex] += this->sBias[nOutputIndex];
		}

		for(auto &nOutput : this->pNextLayer->sOutput)
			nOutput = 1.f / (1.f + std::exp(-nOutput));		//Sigmoid
	}

	void Layer::calcOutputAll()
	{
		this->calcOutput();

		if(this->pNextLayer)
			this->pNextLayer->calcOutputAll();
	}

	void Layer::calcOutputAll(const std::vector<float> &sInputVector)
	{
		for(uint32_t nIndex = 0u, nCount = this->sMatrix.size() ; nIndex < nCount ; ++nIndex)
			this->sOutput[nIndex] = sInputVector[nIndex];

		this->calcOutput();

		if(this->pNextLayer)
			this->pNextLayer->calcOutputAll();
	}

	float Layer::calcError(const std::vector<float> &sDesiredOutputVector)
	{
		float nError = .0f;

		for(uint32_t nIndex = 0u, nCount = this->sOutput.size() ; nIndex < nCount ; ++nIndex)
			nError += std::pow(sDesiredOutputVector[nIndex] - this->sOutput[nIndex], 2);

		return nError * .5f;
	}

	void Layer::clearError()
	{
		for(auto &sError : this->sErrorMatrix)
			memset(sError.data(), 0, sizeof(float) * sError.size());
	}

	void Layer::accrueError(const std::vector<float> &sDesiredOutputVector)
	{
		for(uint32_t nNodeIndex = 0u, nNodeCount = this->sMatrix.size() ; nNodeIndex < nNodeCount ; ++nNodeIndex)
		{
			float nDiff = this->sErrorMatrix.back()[nNodeIndex] = this->sOutput[nNodeIndex] - sDesiredOutputVector[nNodeIndex];		//Cross entropy cost

			for(uint32_t nPrevNodeIndex = 0u, nPrevNodeCount = this->pPrevLayer->sMatrix.size() ; nPrevNodeIndex < nPrevNodeCount ; ++nPrevNodeIndex)
				this->sErrorMatrix[nPrevNodeIndex][nNodeIndex] += this->pPrevLayer->sOutput[nPrevNodeIndex] * nDiff;
		}
	}

	void Layer::learnToBack(float nLearningRate, float nLambda, float nWeightSum, uint32_t nInputListCount)
	{
		if(!this->pPrevLayer)
			return;

		float nFactor = 1.f / nInputListCount;

		if(this->pNextLayer)
			for(uint32_t nNodeIndex = 0u, nNodeCount = this->sMatrix.size() ; nNodeIndex < nNodeCount ; ++nNodeIndex)
			{
				auto &nDelta = this->sErrorMatrix.back()[nNodeIndex] = .0f;

				for(uint32_t nNextNodeIndex = 0u, nNextNodeCount = this->pNextLayer->sMatrix.size() ; nNextNodeIndex < nNextNodeCount ; ++nNextNodeIndex)
					nDelta += this->sMatrix[nNodeIndex][nNextNodeIndex] * this->pNextLayer->sErrorMatrix.back()[nNextNodeIndex];

				nDelta *= this->sOutput[nNodeIndex] * (1.f - this->sOutput[nNodeIndex]);	//Sigmoid

				for(uint32_t nPrevNodeIndex = 0u, nPrevNodeCount = this->pPrevLayer->sMatrix.size() ; nPrevNodeIndex < nPrevNodeCount ; ++nPrevNodeIndex)
					this->pPrevLayer->sMatrix[nPrevNodeIndex][nNodeIndex] -= nLearningRate * this->pPrevLayer->sOutput[nPrevNodeIndex] * nDelta + nFactor * nWeightSum * nLambda * nLearningRate;		//L2 regularization

				this->pPrevLayer->sBias[nNodeIndex] -= nLearningRate * nDelta;
			}
		else
			for(uint32_t nNodeIndex = 0u, nNodeCount = this->sMatrix.size() ; nNodeIndex < nNodeCount ; ++nNodeIndex)
			{
				for(uint32_t nPrevNodeIndex = 0u, nPrevNodeCount = this->pPrevLayer->sMatrix.size() ; nPrevNodeIndex < nPrevNodeCount ; ++nPrevNodeIndex)
					this->pPrevLayer->sMatrix[nPrevNodeIndex][nNodeIndex] -= nFactor * nLearningRate * this->sErrorMatrix[nPrevNodeIndex][nNodeIndex] + nFactor * nWeightSum * nLambda * nLearningRate;		//L2 regularization

				this->pPrevLayer->sBias[nNodeIndex] -= nFactor * nLearningRate * this->sErrorMatrix.back()[nNodeIndex];
			}
	}

	void Layer::learnToBackAll(float nLearningRate, float nLambda, float nWeightSum, uint32_t nInputListCount)
	{
		this->learnToBack(nLearningRate, nLambda, nWeightSum, nInputListCount);

		if(this->pPrevLayer)
			this->pPrevLayer->learnToBackAll(nLearningRate, nLambda, nWeightSum, nInputListCount);
	}

	void Layer::learnToBackAll(float nLearningRate, float nLambda, uint32_t nInputListCount)
	{
		float nWeightSum = .0f;
		for(auto pLayer = this->pPrevLayer ; pLayer ; pLayer = pLayer->pPrevLayer)
			for(const auto &sFactor : pLayer->sMatrix)
				for(auto nFactor : sFactor)
					nWeightSum += nFactor;

		//wprintf_s(L"SUM : %f\n", nWeightSum);

		this->learnToBack(nLearningRate, nLambda, nWeightSum, nInputListCount);

		if(this->pPrevLayer)
			this->pPrevLayer->learnToBackAll(nLearningRate, nLambda, nWeightSum, nInputListCount);
	}

	void Layer::learnToBackAll(Layer &sInputLayer, Layer &sOutputLayer, float nLearningRate, float nLambda, uint32_t nEpochCount, const std::vector<std::vector<float>> &sInputVectorList, const std::vector<std::vector<float>> &sDesiredOutputVectorList)
	{
		uint32_t nInputCount = sInputVectorList.size();

		for(uint32_t nEpoch = 0u ; nEpoch < nEpochCount ; ++nEpoch)
		{
			//wprintf_s(L"Epoch %u...", nEpoch + 1u);
			sOutputLayer.clearError();

			float nError = .0f;

			for(uint32_t nInputIndex = 0u ; nInputIndex < nInputCount ; ++nInputIndex)
			{
				sInputLayer.calcOutputAll(sInputVectorList[nInputIndex]);
				sOutputLayer.accrueError(sDesiredOutputVectorList[nInputIndex]);

				nError += sOutputLayer.calcError(sDesiredOutputVectorList[nInputIndex]);
			}

			//wprintf_s(L"Done w/ error %f.\n", nError / nInputCount);
			sOutputLayer.learnToBackAll(nLearningRate, nLambda, nInputCount);
		}
	}
}