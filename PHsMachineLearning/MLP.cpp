
/*
	2017.02.02
	Created by AcrylicShrimp.
*/

#include "MLP.h"

namespace PHs::MachineLearning
{
	Layer &MLP::addLayerAtFirst(const std::wstring &sLayerName, uint32_t nNodeCount)
	{
		auto &sLayer = this->sLayerMap.emplace(sLayerName, nNodeCount).first->second;
		this->sNameMap.emplace(&sLayer, sLayerName);

		if(this->pInputLayer)
		{
			sLayer.linkLayer(*this->pInputLayer);
			this->pInputLayer = &sLayer;
			this->sInputLayerName = sLayerName;
		}
		else
		{
			this->pOutputLayer = this->pInputLayer = &sLayer;
			this->sOutputLayerName = this->sInputLayerName = sLayerName;
		}

		return sLayer;
	}

	void MLP::addLayerAtFirst(const std::wstring &sLayerName, const Layer &sLayer)
	{
		auto &sNewLayer = this->sLayerMap.emplace(sLayerName, sLayer).first->second;
		this->sNameMap.emplace(&sNewLayer, sLayerName);

		if(this->pInputLayer)
		{
			sNewLayer.linkLayer(*this->pInputLayer);
			this->pInputLayer = &sNewLayer;
			this->sInputLayerName = sLayerName;
		}
		else
		{
			this->pOutputLayer = this->pInputLayer = &sNewLayer;
			this->sOutputLayerName = this->sInputLayerName = sLayerName;
		}
	}

	Layer &MLP::addLayerAtLast(const std::wstring &sLayerName, uint32_t nNodeCount)
	{
		auto &sLayer = this->sLayerMap.emplace(sLayerName, nNodeCount).first->second;
		this->sNameMap.emplace(&sLayer, sLayerName);

		if(this->pOutputLayer)
		{
			this->pOutputLayer->linkLayer(sLayer);
			this->pOutputLayer = &sLayer;
			this->sOutputLayerName = sLayerName;
		}
		else
		{
			this->pInputLayer = this->pOutputLayer = &sLayer;
			this->sInputLayerName = this->sOutputLayerName = sLayerName;
		}

		return sLayer;
	}

	void MLP::addLayerAtLast(const std::wstring &sLayerName, const Layer &sLayer)
	{
		auto &sNewLayer = this->sLayerMap.emplace(sLayerName, sLayer).first->second;
		this->sNameMap.emplace(&sNewLayer, sLayerName);

		if(this->pOutputLayer)
		{
			this->pOutputLayer->linkLayer(sNewLayer);
			this->pOutputLayer = &sNewLayer;
			this->sOutputLayerName = sLayerName;
		}
		else
		{
			this->pInputLayer = this->pOutputLayer = &sNewLayer;
			this->sInputLayerName = this->sOutputLayerName = sLayerName;
		}
	}

	void MLP::writeToFile(std::ofstream &sOutput)
	{
		auto fWriteUnsignedInt = [&sOutput](uint32_t nUnsignedInt)
		{
			union
			{
				uint32_t nUnsignedInt;
				uint8_t vByte[sizeof(uint32_t)];
			}sU2B;

			sU2B.nUnsignedInt = nUnsignedInt;
			sOutput.write(reinterpret_cast <char *>(sU2B.vByte), sizeof(sU2B.vByte));
		};
		auto fWriteFloat = [&sOutput](float nFloat)
		{
			union
			{
				float nFloat;
				uint8_t vByte[sizeof(float)];
			}sF2B;

			sF2B.nFloat = nFloat;
			sOutput.write(reinterpret_cast <char *>(sF2B.vByte), sizeof(sF2B.vByte));
		};
		auto fWriteWideStringToBinary = [fWriteUnsignedInt, &sOutput](const std::wstring &sWideString)
		{
			union
			{
				wchar_t nWideChar;
				uint8_t vByte[sizeof(wchar_t)];
			}sW2B;

			fWriteUnsignedInt(sWideString.length());

			for(auto nWideChar : sWideString)
			{
				sW2B.nWideChar = nWideChar;
				sOutput.write(reinterpret_cast<char *>(sW2B.vByte), sizeof(sW2B.vByte));
			}
		};
		auto fWriteLayerToBinary = [fWriteUnsignedInt, fWriteFloat, &sOutput](const Layer &sLayer)
		{
			fWriteUnsignedInt(sLayer.getMatrix().size());
			fWriteUnsignedInt(sLayer.getBias().size());

			for(const auto &sFactor : sLayer.getMatrix())
				for(auto nFactor : sFactor)
					fWriteFloat(nFactor);

			for(auto nFactor : sLayer.getBias())
				fWriteFloat(nFactor);
		};

		if(this->pInputLayer)
		{
			sOutput << static_cast<uint8_t>(1u);
			fWriteWideStringToBinary(this->sInputLayerName);
			fWriteWideStringToBinary(this->sOutputLayerName);
		}
		else
			sOutput << static_cast<uint8_t>(0u);

		fWriteUnsignedInt(this->sLayerMap.size());

		for(const auto &sPair : this->sLayerMap)
		{
			fWriteWideStringToBinary(sPair.first);
			fWriteLayerToBinary(sPair.second);
		}

		for(const auto &sPair : this->sLayerMap)
		{
			if(sPair.second.getPrevLayer())
			{
				sOutput << static_cast<uint8_t>(1u);
				fWriteWideStringToBinary(this->sNameMap[sPair.second.getPrevLayer()]);
			}
			else
				sOutput << static_cast<uint8_t>(0u);

			if(sPair.second.getNextLayer())
			{
				sOutput << static_cast<uint8_t>(1u);
				fWriteWideStringToBinary(this->sNameMap[sPair.second.getNextLayer()]);
			}
			else
				sOutput << static_cast<uint8_t>(0u);
		}
	}

	void MLP::readFromFile(std::ifstream &sInput)
	{
		this->pInputLayer = this->pOutputLayer = nullptr;
		this->sInputLayerName.clear();
		this->sOutputLayerName.clear();
		this->sLayerMap.clear();
		this->sNameMap.clear();

		auto fReadUnsignedInt = [&sInput]()
		{
			uint32_t nValue;
			sInput.read(reinterpret_cast <char *>(&nValue), sizeof(nValue));
			return nValue;
		};
		auto fReadFloat = [&sInput]()
		{
			float nValue;
			sInput.read(reinterpret_cast <char *>(&nValue), sizeof(nValue));
			return nValue;
		};
		auto fReadWideStringFromBinary = [fReadUnsignedInt, &sInput]()
		{
			union
			{
				wchar_t nWideChar;
				uint8_t vByte[sizeof(wchar_t)];
			}sB2W;

			std::wstring sWideString(fReadUnsignedInt(), L'\0');

			for(auto &nWideChar : sWideString)
			{
				sInput.read(reinterpret_cast <char *>(sB2W.vByte), sizeof(sB2W.vByte));
				nWideChar = sB2W.nWideChar;
			}

			return sWideString;
		};
		auto fReadLayerFromBinary = [fReadUnsignedInt, fReadFloat, &sInput]()
		{
			uint32_t nNodeCount = fReadUnsignedInt();
			uint32_t nBiasCount = fReadUnsignedInt();

			Layer sLayer{nNodeCount};

			for(auto &sFactor : sLayer.getMatrix())
			{
				sFactor.resize(nBiasCount);
				for(auto &nFactor : sFactor)
					nFactor = fReadFloat();
			}

			sLayer.getBias().resize(nBiasCount);
			for(auto &nFactor : sLayer.getBias())
				nFactor = fReadFloat();

			return sLayer;
		};

		if(sInput.get())
		{
			this->sInputLayerName = fReadWideStringFromBinary();
			this->sOutputLayerName = fReadWideStringFromBinary();
		}

		for(uint32_t nCount = 0u, nMapSize = fReadUnsignedInt() ; nCount < nMapSize ; ++nCount)
		{
			auto sLayerName = fReadWideStringFromBinary();
			this->sLayerMap.emplace(std::move(sLayerName), fReadLayerFromBinary());
		}

		for(auto &sPair : this->sLayerMap)
		{
			if(sInput.get())
				if(!sPair.second.getPrevLayer())
					this->sLayerMap[fReadWideStringFromBinary()].linkLayerWithoutModify(sPair.second);
				else
					fReadWideStringFromBinary();

			if(sInput.get())
				if(!sPair.second.getNextLayer())
					sPair.second.linkLayerWithoutModify(this->sLayerMap[fReadWideStringFromBinary()]);
				else
					fReadWideStringFromBinary();

			this->sNameMap.emplace(&sPair.second, sPair.first);
		}

		if(this->sInputLayerName.size())
		{
			this->pInputLayer = &this->sLayerMap[this->sInputLayerName];
			this->pOutputLayer = &this->sLayerMap[this->sOutputLayerName];
		}
	}
}