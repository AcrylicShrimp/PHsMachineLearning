
/*
	2017.02.02
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_PHS_MACHINELEARNING_MLP_H

#define _CLASS_PHS_MACHINELEARNING_MLP_H

#include "Layer.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>

namespace PHs::MachineLearning
{
	class MLP
	{
	private:
		Layer *pInputLayer{nullptr};
		Layer *pOutputLayer{nullptr};
		std::wstring sInputLayerName;
		std::wstring sOutputLayerName;
		std::map<std::wstring, Layer> sLayerMap;
		std::unordered_map<Layer *, std::wstring> sNameMap;
		
	public:
		MLP() = default;
		MLP(const MLP &rSrc) = delete;
		MLP(MLP &&rSrc) = delete;
		~MLP() = default;
		
	public:
		Layer &addLayerAtFirst(const std::wstring &sLayerName, uint32_t nNodeCount);
		void addLayerAtFirst(const std::wstring &sLayerName, const Layer &sLayer);
		Layer &addLayerAtLast(const std::wstring &sLayerName, uint32_t nNodeCount);
		void addLayerAtLast(const std::wstring &sLayerName, const Layer &sLayer);
		void writeToFile(std::ofstream &sOutput);
		void readFromFile(std::ifstream &sInput);

		inline Layer *getInputLayer();
		inline const Layer *getInputLayer() const;
		inline Layer *getOutputLayer();
		inline const Layer *getOutputLayer() const;
		inline Layer *getLayer(const std::wstring &sLayerName);
		inline const Layer *getLayer(const std::wstring &sLayerName) const;
	};

	inline Layer *MLP::getInputLayer()
	{
		return this->pInputLayer;
	}
	
	inline const Layer *MLP::getInputLayer() const
	{
		return this->pInputLayer;
	}

	inline Layer *MLP::getOutputLayer()
	{
		return this->pOutputLayer;
	}

	inline const Layer *MLP::getOutputLayer() const
	{
		return this->pOutputLayer;
	}

	inline Layer *MLP::getLayer(const std::wstring &sLayerName)
	{
		auto iIndex = this->sLayerMap.find(sLayerName);

		return iIndex != this->sLayerMap.end() ? &iIndex->second : nullptr;
	}

	inline const Layer *MLP::getLayer(const std::wstring &sLayerName) const
	{
		auto iIndex = this->sLayerMap.find(sLayerName);

		return iIndex != this->sLayerMap.cend() ? &iIndex->second : nullptr;
	}
}

#endif