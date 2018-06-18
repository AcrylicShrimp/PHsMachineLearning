
/*
	2016.09.15
	Created by PHJ.
*/

#ifndef _RUN_CPP

#define _RUN_CPP

/*
	TODO : Place your code here
*/

#include "Layer.h"
#include "MLP.h"
#include "PHsMultiLayeredPerceptron.h"

#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <io.h>
#include <vector>

using namespace PHs::PHsMachineLearning;
using namespace PHs::MachineLearning;

using namespace std;

int main()
{
	_setmode(_fileno(stdin), _O_U16TEXT);
	_setmode(_fileno(stderr), _O_U16TEXT);
	_setmode(_fileno(stdout), _O_U16TEXT);

	vector<vector<float>> sInput;
#pragma region InputData
	sInput.push_back({0.787f, 0.186f, 0.455f, 0.278f, 0.283f, 0.576f, 0.42f, 0.245f, 0.495f, 0.292f, 0.455f, 0.85f, 0.54f});
	sInput.push_back({0.439f, 0.555f, 0.535f, 0.562f, 0.391f, 0.248f, 0.181f, 0.075f, 0.136f, 0.317f, 0.244f, 0.007f, 0.23f});
	sInput.push_back({0.65f, 0.47f, 0.674f, 0.691f, 0.576f, 0.145f, 0.259f, 0.17f, 0.265f, 0.625f, 0.089f, 0.011f, 0.158f});
	sInput.push_back({0.532f, 1.f, 0.412f, 0.562f, 0.174f, 0.566f, 0.487f, 0.321f, 0.505f, 0.113f, 0.203f, 0.67f, 0.073f});
	sInput.push_back({0.479f, 0.17f, 0.62f, 0.371f, 0.272f, 0.517f, 0.428f, 0.245f, 0.331f, 0.226f, 0.496f, 0.864f, 0.526f});
	sInput.push_back({0.879f, 0.239f, 0.61f, 0.32f, 0.467f, 0.99f, 0.665f, 0.208f, 0.558f, 0.556f, 0.309f, 0.799f, 0.857f});
	sInput.push_back({0.687f, 0.466f, 0.642f, 0.237f, 0.5f, 0.593f, 0.568f, 0.075f, 0.394f, 0.326f, 0.39f, 0.766f, 0.404f});
	sInput.push_back({0.445f, 0.2f, 0.492f, 0.613f, 0.152f, 0.138f, 0.3f, 0.66f, 0.385f, 0.172f, 0.325f, 0.421f, 0.15f});
	sInput.push_back({0.721f, 0.229f, 0.706f, 0.335f, 0.489f, 0.697f, 0.517f, 0.491f, 0.401f, 0.428f, 0.528f, 0.608f, 0.782f});
	sInput.push_back({0.582f, 0.366f, 0.807f, 0.536f, 0.522f, 0.628f, 0.496f, 0.491f, 0.445f, 0.259f, 0.455f, 0.608f, 0.326f});
	sInput.push_back({0.221f, 0.706f, 0.551f, 0.536f, 0.13f, 0.648f, 0.568f, 0.151f, 0.789f, 0.13f, 0.22f, 0.868f, 0.073f});
	sInput.push_back({0.739f, 0.668f, 0.545f, 0.459f, 0.207f, 0.283f, 0.103f, 0.66f, 0.363f, 0.66f, 0.073f, 0.136f, 0.144f});
	sInput.push_back({0.395f, 0.943f, 0.684f, 0.742f, 0.283f, 0.279f, 0.055f, 0.943f, 0.218f, 0.317f, 0.276f, 0.154f, 0.169f});
	sInput.push_back({0.276f, 0.077f, 0.615f, 0.691f, 0.087f, 0.352f, 0.262f, 0.509f, 0.312f, 0.078f, 0.675f, 0.531f, 0.251f});
	sInput.push_back({0.837f, 0.652f, 0.578f, 0.428f, 0.446f, 0.645f, 0.487f, 0.321f, 0.265f, 0.338f, 0.317f, 0.755f, 0.572f});
	sInput.push_back({0.439f, 0.619f, 0.556f, 0.639f, 0.337f, 0.638f, 0.466f, 0.566f, 0.486f, 0.11f, 0.577f, 0.681f, 0.132f});
	sInput.push_back({0.75f, 0.227f, 0.658f, 0.227f, 0.337f, 0.783f, 0.679f, 0.075f, 0.407f, 0.354f, 0.325f, 0.839f, 0.583f});
	sInput.push_back({0.832f, 0.168f, 0.599f, 0.304f, 0.413f, 0.8f, 0.757f, 0.358f, 0.457f, 0.633f, 0.61f, 0.568f, 1.f});
	sInput.push_back({0.321f, 0.621f, 0.449f, 0.407f, 0.457f, 0.138f, 0.093f, 0.302f, 0.23f, 0.591f, 0.138f, 0.267f, 0.412f});
	sInput.push_back({0.508f, 0.536f, 0.529f, 0.407f, 0.391f, 0.141f, 0.076f, 0.509f, 0.167f, 0.341f, 0.163f, 0.176f, 0.283f});
	sInput.push_back({0.1f, 0.f, 0.61f, 0.536f, 0.196f, 0.517f, 0.352f, 0.547f, 0.325f, 0.154f, 0.504f, 0.381f, 0.111f});
	sInput.push_back({0.871f, 0.186f, 0.717f, 0.742f, 0.304f, 0.628f, 0.205f, 0.755f, 0.722f, 1.f, 0.073f, 0.253f, 0.272f});
	sInput.push_back({0.321f, 0.196f, 0.406f, 0.433f, 0.109f, 0.231f, 0.357f, 0.453f, 0.385f, 0.181f, 0.423f, 0.696f, 0.165f});
	sInput.push_back({0.592f, 0.178f, 0.791f, 0.253f, 0.435f, 0.559f, 0.494f, 0.396f, 0.3f, 0.283f, 0.496f, 0.553f, 0.429f});
	sInput.push_back({0.297f, 0.172f, 0.508f, 0.629f, 0.217f, 0.276f, 0.285f, 0.566f, 0.363f, 0.1f, 0.691f, 0.363f, 0.155f});
	sInput.push_back({0.613f, 0.36f, 0.529f, 0.485f, 0.207f, 0.145f, 0.034f, 0.453f, 0.073f, 0.369f, 0.179f, 0.44f, 0.358f});
	sInput.push_back({0.213f, 0.03f, 0.652f, 0.381f, 0.261f, 0.421f, 0.395f, 0.17f, 0.612f, 0.151f, 0.252f, 0.663f, 0.173f});
	sInput.push_back({0.432f, 0.047f, 0.471f, 0.381f, 0.315f, 0.421f, 0.338f, 0.321f, 0.331f, 0.114f, 0.61f, 0.692f, 0.123f});
	sInput.push_back({0.332f, 0.172f, 0.455f, 0.505f, 0.359f, 0.041f, 0.143f, 0.453f, 0.331f, 0.151f, 0.346f, 0.201f, 0.422f});
	sInput.push_back({0.332f, 0.48f, 0.455f, 0.381f, 0.196f, 0.645f, 0.559f, 0.604f, 0.757f, 0.087f, 0.764f, 0.571f, 0.091f});
	sInput.push_back({1.f, 0.178f, 0.433f, 0.175f, 0.293f, 0.628f, 0.557f, 0.302f, 0.495f, 0.334f, 0.488f, 0.579f, 0.547f});
	sInput.push_back({0.342f, 0.049f, 0.316f, 0.216f, 0.717f, 0.317f, 0.319f, 0.415f, 0.741f, 0.181f, 0.472f, 0.381f, 0.337f});
	sInput.push_back({0.75f, 0.85f, 0.465f, 0.485f, 0.109f, 0.f, 0.f, 0.509f, 0.085f, 0.309f, 0.081f, 0.022f, 0.098f});
	sInput.push_back({0.342f, 0.071f, 0.492f, 0.278f, 0.337f, 0.369f, 0.158f, 0.943f, 0.f, 0.17f, 0.626f, 0.147f, 0.287f});
	sInput.push_back({0.353f, 0.093f, 0.642f, 0.387f, 0.304f, 0.497f, 0.487f, 0.453f, 0.527f, 0.283f, 0.577f, 0.377f, 0.285f});
	sInput.push_back({0.745f, 0.152f, 0.701f, 0.742f, 0.174f, 0.679f, 0.532f, 0.151f, 0.461f, 0.179f, 0.715f, 0.692f, 0.094f});
	sInput.push_back({0.308f, 0.453f, 0.513f, 0.433f, 0.283f, 0.093f, 0.032f, 0.509f, 0.101f, 0.36f, 0.146f, 0.205f, 0.165f});
	sInput.push_back({0.321f, 0.787f, 0.631f, 0.536f, 0.207f, 0.138f, 0.027f, 0.755f, 0.123f, 0.219f, 0.22f, 0.f, 0.315f});
	sInput.push_back({0.353f, 0.04f, 0.f, 0.f, 0.196f, 0.345f, 0.049f, 0.283f, 0.003f, 0.057f, 0.463f, 0.201f, 0.173f});
	sInput.push_back({0.666f, 0.196f, 0.588f, 0.51f, 0.5f, 0.683f, 0.515f, 0.132f, 0.644f, 0.424f, 0.407f, 0.645f, 0.601f});
	sInput.push_back({0.711f, 0.15f, 0.717f, 0.613f, 0.337f, 0.697f, 0.614f, 0.302f, 0.621f, 0.377f, 0.577f, 0.527f, 0.718f});
	sInput.push_back({0.808f, 0.253f, 0.556f, 0.423f, 0.359f, 0.61f, 0.544f, 0.358f, 0.621f, 0.42f, 0.48f, 0.542f, 0.558f});
	sInput.push_back({0.671f, 0.364f, 0.711f, 0.716f, 0.38f, 0.197f, 0.105f, 0.491f, 0.356f, 0.63f, 0.211f, 0.194f, 0.337f});
	sInput.push_back({0.708f, 0.136f, 0.61f, 0.314f, 0.413f, 0.834f, 0.703f, 0.113f, 0.514f, 0.471f, 0.333f, 0.586f, 0.718f});
	sInput.push_back({0.595f, 0.243f, 0.706f, 0.32f, 0.348f, 0.697f, 0.61f, 0.34f, 0.394f, 0.403f, 0.48f, 0.575f, 0.708f});
	sInput.push_back({0.366f, 0.172f, 0.444f, 0.613f, 0.413f, 0.352f, 0.369f, 0.396f, 0.379f, 0.067f, 0.472f, 0.619f, 0.048f});
	sInput.push_back({0.603f, 0.494f, 0.545f, 0.562f, 0.239f, 0.328f, 0.089f, 0.604f, 0.265f, 0.609f, 0.057f, 0.128f, 0.265f});
	sInput.push_back({0.766f, 0.196f, 0.487f, 0.351f, 0.413f, 0.655f, 0.675f, 0.358f, 0.527f, 0.65f, 0.52f, 0.67f, 0.7f});
	sInput.push_back({0.626f, 0.613f, 0.406f, 0.423f, 0.217f, 0.507f, 0.494f, 0.264f, 0.338f, 0.256f, 0.35f, 0.634f, 0.54f});
	sInput.push_back({0.655f, 0.48f, 0.727f, 0.665f, 0.293f, 0.197f, 0.038f, 0.698f, 0.044f, 0.262f, 0.333f, 0.289f, 0.173f});
	sInput.push_back({0.276f, 0.265f, 0.182f, 0.356f, 0.293f, 0.431f, 0.386f, 0.245f, 0.312f, 0.172f, 0.642f, 0.619f, 0.308f});
	sInput.push_back({0.208f, 0.194f, 0.278f, 0.459f, 0.174f, 0.524f, 0.274f, 0.453f, 0.319f, 0.067f, 0.374f, 0.429f, 0.098f});
	sInput.push_back({0.5f, 0.605f, 0.69f, 0.412f, 0.348f, 0.493f, 0.437f, 0.226f, 0.495f, 0.275f, 0.447f, 0.824f, 0.351f});
	sInput.push_back({0.276f, 0.128f, 0.61f, 0.613f, 0.152f, 0.545f, 0.411f, 0.566f, 0.199f, 0.138f, 0.366f, 0.703f, 0.076f});
	sInput.push_back({0.705f, 0.97f, 0.583f, 0.51f, 0.272f, 0.241f, 0.057f, 0.736f, 0.205f, 0.548f, 0.13f, 0.172f, 0.33f});
	sInput.push_back({0.716f, 0.196f, 0.561f, 0.278f, 0.207f, 0.559f, 0.511f, 0.302f, 0.442f, 0.369f, 0.545f, 0.597f, 0.743f});
	sInput.push_back({0.f, 0.152f, 0.449f, 0.562f, 0.163f, 0.51f, 0.386f, 0.736f, 0.505f, 0.053f, 1.f, 0.586f, 0.092f});
	sInput.push_back({0.608f, 0.04f, 0.535f, 0.33f, 0.435f, 0.534f, 0.203f, 0.792f, 0.003f, 0.161f, 0.439f, 0.242f, 0.337f});
	sInput.push_back({0.255f, 0.152f, 0.567f, 0.588f, 0.174f, 0.162f, 0.192f, 0.698f, 0.385f, 0.198f, 0.463f, 0.505f, 0.123f});
	sInput.push_back({0.276f, 0.215f, 0.513f, 0.407f, 0.12f, 0.214f, 0.245f, 0.736f, 0.388f, 0.096f, 0.488f, 0.366f, 0.144f});
	sInput.push_back({0.813f, 0.146f, 0.513f, 0.32f, 0.272f, 0.421f, 0.441f, 0.245f, 0.366f, 0.317f, 0.561f, 0.568f, 0.715f});
	sInput.push_back({0.192f, 0.383f, 0.834f, 0.485f, 0.359f, 0.266f, 0.357f, 0.887f, 0.202f, 0.215f, 0.61f, 0.451f, 0.235f});
	sInput.push_back({0.516f, 0.184f, 0.663f, 1.f, 0.75f, 0.8f, 0.538f, 0.151f, 0.489f, 0.177f, 0.675f, 0.817f, 0.504f});
	sInput.push_back({0.65f, 0.211f, 0.668f, 0.485f, 0.283f, 0.534f, 0.479f, 0.283f, 0.394f, 0.191f, 0.52f, 0.934f, 0.404f});
	sInput.push_back({0.745f, 0.121f, 0.487f, 0.278f, 0.304f, 0.69f, 0.593f, 0.17f, 0.454f, 0.507f, 0.431f, 0.835f, 0.547f});
	sInput.push_back({0.266f, 0.704f, 0.545f, 0.588f, 0.109f, 0.386f, 0.297f, 0.547f, 0.297f, 0.113f, 0.252f, 0.476f, 0.215f});
	sInput.push_back({0.353f, 0.065f, 0.396f, 0.407f, 0.196f, 0.876f, 0.719f, 0.208f, 0.486f, 0.275f, 0.455f, 0.549f, 0.272f});
	sInput.push_back({0.276f, 0.117f, 0.503f, 0.67f, 0.f, 0.421f, 0.264f, 0.547f, 0.306f, 0.039f, 0.48f, 0.711f, 0.248f});
	sInput.push_back({0.684f, 0.211f, 0.717f, 0.34f, 0.457f, 0.645f, 0.542f, 0.321f, 0.331f, 0.514f, 0.65f, 0.59f, 0.736f});
	sInput.push_back({0.624f, 0.763f, 0.802f, 0.742f, 0.457f, 0.345f, 0.131f, 0.264f, 0.221f, 0.616f, 0.154f, 0.238f, 0.251f});
	sInput.push_back({0.139f, 0.259f, 1.f, 0.923f, 0.533f, 0.759f, 1.f, 0.642f, 0.461f, 0.403f, 0.366f, 0.886f, 0.133f});
	sInput.push_back({0.374f, 0.453f, 0.684f, 0.845f, 0.293f, 0.317f, 0.051f, 0.943f, 0.23f, 0.531f, 0.154f, 0.168f, 0.429f});
	sInput.push_back({0.161f, 0.261f, 0.588f, 0.567f, 0.152f, 0.334f, 0.285f, 0.66f, 0.297f, 0.13f, 0.423f, 0.542f, 0.287f});
	sInput.push_back({0.532f, 0.617f, 0.513f, 0.613f, 0.163f, 0.231f, 0.264f, 0.906f, 0.382f, 0.3f, 0.293f, 0.271f, 0.169f});
	sInput.push_back({0.345f, 0.338f, 0.588f, 0.536f, 0.304f, 0.545f, 0.373f, 0.396f, 0.284f, 0.13f, 0.26f, 0.773f, 0.114f});
	sInput.push_back({0.532f, 0.196f, 0.364f, 0.093f, 0.239f, 0.6f, 0.618f, 0.075f, 0.789f, 0.505f, 0.52f, 0.601f, 0.622f});
	sInput.push_back({0.808f, 0.281f, 0.503f, 0.381f, 0.38f, 0.679f, 0.629f, 0.17f, 0.621f, 0.381f, 0.626f, 0.696f, 0.879f});
	sInput.push_back({0.621f, 0.204f, 0.674f, 0.284f, 0.25f, 0.645f, 0.549f, 0.396f, 0.328f, 0.3f, 0.358f, 0.714f, 0.654f});
	sInput.push_back({0.563f, 0.366f, 0.54f, 0.485f, 0.543f, 0.231f, 0.072f, 0.755f, 0.331f, 0.684f, 0.098f, 0.128f, 0.401f});
	sInput.push_back({0.311f, 0.089f, 0.209f, 0.32f, 0.88f, 0.3f, 0.198f, 0.019f, 0.659f, 0.134f, 0.65f, 0.659f, 0.314f});
	sInput.push_back({0.458f, 0.532f, 0.332f, 0.278f, 0.109f, 0.224f, 0.192f, 0.566f, 0.132f, 0.181f, 0.179f, 0.311f, 0.067f});
	sInput.push_back({0.797f, 0.176f, 0.492f, 0.278f, 0.609f, 0.697f, 0.597f, 0.208f, 0.533f, 0.373f, 0.496f, 0.894f, 0.358f});
	sInput.push_back({0.458f, 0.326f, 0.492f, 0.459f, 0.174f, 0.141f, 0.036f, 0.66f, 0.073f, 0.735f, 0.073f, 0.132f, 0.137f});
	sInput.push_back({0.389f, 0.099f, 0.476f, 0.356f, 0.163f, 0.352f, 0.051f, 0.887f, 0.265f, 0.356f, 0.22f, 0.088f, 0.265f});
	sInput.push_back({0.155f, 0.247f, 0.492f, 0.381f, 0.304f, 0.703f, 0.405f, 0.075f, 0.297f, 0.168f, 0.553f, 0.619f, 0.048f});
	sInput.push_back({0.884f, 0.223f, 0.583f, 0.206f, 0.283f, 0.524f, 0.46f, 0.321f, 0.495f, 0.339f, 0.439f, 0.846f, 0.722f});
	sInput.push_back({0.666f, 0.192f, 0.508f, 0.289f, 0.511f, 0.748f, 0.622f, 0.396f, 0.609f, 0.414f, 0.382f, 0.773f, 0.369f});
	sInput.push_back({0.5f, 0.409f, 0.717f, 0.536f, 0.283f, 0.193f, 0.034f, 0.755f, 0.107f, 0.283f, 0.236f, 0.381f, 0.23f});
	sInput.push_back({0.839f, 0.642f, 0.615f, 0.134f, 0.63f, 0.697f, 0.57f, 0.132f, 0.527f, 0.326f, 0.333f, 0.828f, 0.344f});
	sInput.push_back({0.471f, 0.52f, 0.503f, 0.459f, 0.196f, 0.172f, 0.068f, 0.509f, 0.177f, 0.766f, 0.195f, 0.176f, 0.29f});
	sInput.push_back({0.484f, 0.765f, 0.599f, 0.562f, 0.174f, 0.248f, 0.065f, 0.642f, 0.142f, 0.544f, 0.049f, 0.216f, 0.248f});
	sInput.push_back({0.671f, 0.182f, 0.535f, 0.438f, 0.391f, 0.648f, 0.601f, 0.17f, 0.486f, 0.48f, 0.496f, 0.59f, 0.882f});
	sInput.push_back({0.624f, 0.626f, 0.599f, 0.639f, 0.348f, 0.283f, 0.086f, 0.566f, 0.315f, 0.514f, 0.179f, 0.106f, 0.337f});
	sInput.push_back({0.682f, 0.832f, 0.529f, 0.485f, 0.239f, 0.352f, 0.097f, 0.642f, 0.192f, 0.266f, 0.35f, 0.286f, 0.194f});
	sInput.push_back({0.589f, 0.7f, 0.481f, 0.485f, 0.543f, 0.21f, 0.074f, 0.566f, 0.297f, 0.761f, 0.089f, 0.106f, 0.397f});
	sInput.push_back({0.547f, 0.229f, 0.743f, 0.768f, 0.5f, 0.421f, 0.198f, 0.245f, 0.363f, 0.497f, 0.106f, 0.022f, 0.105f});
	sInput.push_back({0.113f, 0.593f, 0.246f, 0.459f, 0.402f, 0.759f, 0.473f, 0.208f, 1.f, 0.138f, 0.22f, 0.564f, 0.203f});
	sInput.push_back({0.579f, 0.506f, 0.492f, 0.407f, 0.304f, 0.283f, 0.103f, 0.906f, 0.461f, 0.788f, 0.065f, 0.088f, 0.283f});
	sInput.push_back({0.255f, 0.036f, 0.342f, 0.433f, 0.174f, 0.497f, 0.405f, 0.321f, 0.322f, 0.104f, 0.732f, 0.678f, 0.f});
	sInput.push_back({0.153f, 0.121f, 0.717f, 0.485f, 0.261f, 0.607f, 0.544f, 0.302f, 0.656f, 0.117f, 0.39f, 0.729f, 0.287f});
	sInput.push_back({0.379f, 0.154f, 0.449f, 0.433f, 1.f, 0.524f, 0.407f, 0.358f, 0.905f, 0.113f, 0.553f, 0.498f, 0.47f});
	sInput.push_back({0.537f, 0.15f, 0.396f, 0.253f, 0.304f, 0.49f, 0.485f, 0.283f, 0.303f, 0.206f, 0.569f, 0.52f, 0.529f});
	sInput.push_back({0.647f, 0.182f, 0.471f, 0.691f, 0.185f, 0.31f, 0.316f, 0.264f, 0.196f, 0.21f, 0.407f, 0.553f, 0.138f});
	sInput.push_back({0.834f, 0.202f, 0.583f, 0.237f, 0.457f, 0.79f, 0.643f, 0.396f, 0.492f, 0.467f, 0.463f, 0.579f, 0.836f});
	sInput.push_back({0.734f, 0.2f, 0.567f, 0.175f, 0.446f, 1.f, 0.717f, 0.358f, 0.461f, 0.492f, 0.431f, 0.729f, 0.65f});
	sInput.push_back({0.463f, 0.381f, 0.599f, 0.588f, 0.457f, 0.172f, 0.215f, 0.208f, 0.268f, 0.812f, 0.f, 0.073f, 0.144f});
	sInput.push_back({0.571f, 0.206f, 0.417f, 0.031f, 0.326f, 0.576f, 0.511f, 0.245f, 0.274f, 0.265f, 0.463f, 0.78f, 0.551f});
	sInput.push_back({0.166f, 0.225f, 0.299f, 0.278f, 0.293f, 0.217f, 0.259f, 0.396f, 0.233f, 0.215f, 0.61f, 0.319f, 0.107f});
	sInput.push_back({0.353f, 0.176f, 0.503f, 0.716f, 0.196f, 0.428f, 0.445f, 0.509f, 0.47f, 0.072f, 0.333f, 0.553f, 0.046f});
	sInput.push_back({0.563f, 0.879f, 0.513f, 0.588f, 0.25f, 0.262f, 0.061f, 0.906f, 0.36f, 0.565f, 0.098f, 0.077f, 0.319f});
	sInput.push_back({0.724f, 0.399f, 0.503f, 0.588f, 0.217f, 0.128f, 0.072f, 0.528f, 0.196f, 0.708f, 0.179f, 0.15f, 0.24f});
	sInput.push_back({0.313f, 0.109f, 0.31f, 0.433f, 0.239f, 0.476f, 0.359f, 0.491f, 0.527f, 0.121f, 0.309f, 0.641f, 0.024f});
	sInput.push_back({0.482f, 0.121f, 0.513f, 0.381f, 0.565f, 0.183f, 0.192f, 0.151f, 0.167f, 0.241f, 0.228f, 0.007f, 0.251f});
	sInput.push_back({0.332f, 0.132f, 0.332f, 0.278f, 0.163f, 0.541f, 0.456f, 0.302f, 0.429f, 0.138f, 0.61f, 0.538f, 0.107f});
	sInput.push_back({0.163f, 0.184f, 0.674f, 0.794f, 0.196f, 0.324f, 0.268f, 0.509f, 0.293f, 0.113f, 0.715f, 0.711f, 0.203f});
	sInput.push_back({0.476f, 0.439f, 0.668f, 0.691f, 0.337f, 0.462f, 0.055f, 0.755f, 0.126f, 0.311f, 0.333f, 0.322f, 0.223f});
	sInput.push_back({0.332f, 0.413f, 0.46f, 0.381f, 0.196f, 0.507f, 0.403f, 0.226f, 0.498f, 0.074f, 0.545f, 0.744f, 0.009f});
	sInput.push_back({0.842f, 0.192f, 0.572f, 0.258f, 0.62f, 0.628f, 0.574f, 0.283f, 0.593f, 0.372f, 0.455f, 0.971f, 0.561f});
	sInput.push_back({0.697f, 0.215f, 0.535f, 0.34f, 0.37f, 0.497f, 0.496f, 0.547f, 0.492f, 0.218f, 0.61f, 0.586f, 0.508f});
	sInput.push_back({0.705f, 0.221f, 0.535f, 0.309f, 0.337f, 0.562f, 0.536f, 0.264f, 0.404f, 0.215f, 0.512f, 1.f, 0.54f});
	sInput.push_back({0.3f, 0.14f, 0.626f, 0.433f, 0.37f, 0.314f, 0.297f, 0.604f, 0.196f, 0.142f, 0.789f, 0.352f, 0.055f});
	sInput.push_back({0.487f, 0.445f, 0.556f, 0.485f, 0.37f, 0.11f, 0.186f, 0.208f, 0.132f, 0.352f, 0.211f, 0.055f, 0.18f});
	sInput.push_back({0.424f, 0.123f, 0.353f, 0.32f, 0.326f, 0.359f, 0.226f, 0.755f, 0.066f, 0.381f, 0.407f, 0.117f, 0.123f});
	sInput.push_back({0.882f, 0.563f, 0.492f, 0.278f, 0.348f, 0.783f, 0.597f, 0.264f, 0.562f, 0.309f, 0.455f, 0.795f, 0.561f});
	sInput.push_back({0.839f, 0.19f, 0.503f, 0.294f, 0.522f, 0.766f, 0.561f, 0.245f, 0.511f, 0.435f, 0.374f, 0.747f, 0.494f});
	sInput.push_back({0.797f, 0.279f, 0.668f, 0.361f, 0.554f, 0.559f, 0.458f, 0.34f, 0.265f, 0.322f, 0.472f, 0.846f, 0.725f});
	sInput.push_back({0.35f, 0.611f, 0.545f, 0.536f, 0.196f, 0.455f, 0.122f, 0.698f, 0.199f, 0.544f, 0.065f, 0.114f, 0.173f});
	sInput.push_back({0.824f, 0.35f, 0.599f, 0.485f, 0.228f, 0.241f, 0.076f, 0.585f, 0.262f, 0.718f, 0.114f, 0.161f, 0.272f});
	sInput.push_back({0.755f, 0.186f, 0.406f, 0.278f, 0.337f, 0.731f, 0.643f, 0.151f, 0.546f, 0.411f, 0.35f, 0.755f, 0.504f});
	sInput.push_back({0.111f, 0.328f, 0.567f, 0.485f, 0.283f, 0.662f, 0.517f, 0.358f, 0.448f, 0.168f, 0.26f, 0.777f, 0.248f});
	sInput.push_back({0.582f, 0.64f, 0.497f, 0.356f, 0.359f, 0.572f, 0.483f, 0.358f, 0.394f, 0.263f, 0.276f, 0.634f, 0.287f});
	sInput.push_back({0.2f, 0.275f, 0.759f, 0.923f, 0.239f, 0.397f, 0.401f, 0.849f, 0.426f, 0.147f, 0.398f, 0.429f, 0.134f});
	sInput.push_back({0.468f, 0.31f, 0.556f, 0.691f, 0.304f, 0.059f, 0.158f, 0.264f, 0.132f, 0.377f, 0.146f, 0.033f, 0.201f});
	sInput.push_back({0.597f, 0.194f, 0.417f, 0.33f, 0.261f, 0.49f, 0.39f, 0.264f, 0.297f, 0.228f, 0.439f, 0.549f, 0.718f});
	sInput.push_back({0.366f, 0.358f, 0.487f, 0.588f, 0.217f, 0.241f, 0.316f, 1.f, 0.319f, 0.121f, 0.309f, 0.744f, 0.026f});
	sInput.push_back({0.245f, 0.069f, 0.503f, 0.536f, 0.337f, 0.828f, 0.38f, 0.f, 0.391f, 0.165f, 0.415f, 0.681f, 0.434f});
	sInput.push_back({0.413f, 0.34f, 0.449f, 0.407f, 0.261f, 0.221f, 0.068f, 0.943f, 0.167f, 0.497f, 0.203f, 0.114f, 0.297f});
	sInput.push_back({0.645f, 0.211f, 0.561f, 0.51f, 0.326f, 0.593f, 0.557f, 0.245f, 0.457f, 0.326f, 0.455f, 0.806f, 0.458f});
	sInput.push_back({0.526f, 0.032f, 0.187f, 0.278f, 0.174f, 0.334f, 0.357f, 0.208f, 0.331f, 0.283f, 0.577f, 0.443f, 0.081f});
	sInput.push_back({0.408f, 0.109f, 0.396f, 0.485f, 0.359f, 0.172f, 0.051f, 0.755f, 0.312f, 0.539f, 0.081f, 0.103f, 0.258f});
	sInput.push_back({0.389f, 0.196f, 0.332f, 0.51f, 0.163f, 0.421f, 0.333f, 0.358f, 0.338f, 0.142f, 0.455f, 0.842f, 0.281f});
	sInput.push_back({0.737f, 0.164f, 0.674f, 0.485f, 0.489f, 0.679f, 0.646f, 0.509f, 0.413f, 0.454f, 0.528f, 0.476f, 0.608f});
	sInput.push_back({0.532f, 0.204f, 0.396f, 0.33f, 0.402f, 0.697f, 0.561f, 0.283f, 0.511f, 0.321f, 0.325f, 0.762f, 0.433f});
	sInput.push_back({0.882f, 0.223f, 0.545f, 0.072f, 0.348f, 0.8f, 0.696f, 0.302f, 0.804f, 0.531f, 0.585f, 0.634f, 0.905f});
	sInput.push_back({0.653f, 0.209f, 0.69f, 0.433f, 0.435f, 0.472f, 0.462f, 0.302f, 0.356f, 0.249f, 0.504f, 0.586f, 0.583f});
	sInput.push_back({0.816f, 0.664f, 0.738f, 0.716f, 0.283f, 0.369f, 0.089f, 0.811f, 0.297f, 0.676f, 0.106f, 0.121f, 0.201f});
	sInput.push_back({0.637f, 0.585f, 0.663f, 0.639f, 0.446f, 0.248f, 0.122f, 0.566f, 0.331f, 0.802f, 0.301f, 0.106f, 0.297f});
	sInput.push_back({0.445f, 0.211f, 0.449f, 0.423f, 0.174f, 0.421f, 0.462f, 0.245f, 0.429f, 0.224f, 0.553f, 0.685f, 0.311f});
	sInput.push_back({0.368f, 0.156f, 0.497f, 0.562f, 0.174f, 0.607f, 0.593f, 0.491f, 0.429f, 0.227f, 0.171f, 0.575f, 0.053f});
	sInput.push_back({0.532f, 0.18f, 0.636f, 0.381f, 0.304f, 0.507f, 0.441f, 0.302f, 0.325f, 0.253f, 0.52f, 0.454f, 0.59f});
	sInput.push_back({0.561f, 0.559f, 0.422f, 0.536f, 0.348f, 0.179f, 0.044f, 0.566f, 0.281f, 0.232f, 0.098f, 0.15f, 0.394f});
	sInput.push_back({0.213f, 0.425f, 0.465f, 0.381f, 0.457f, 0.255f, 0.207f, 0.566f, 0.17f, 0.117f, 0.39f, 0.458f, 0.158f});
	sInput.push_back({0.353f, 0.085f, 0.299f, 0.464f, 0.087f, 0.39f, 0.35f, 0.264f, 0.199f, 0.29f, 0.52f, 0.81f, 0.165f});
	sInput.push_back({0.747f, 0.229f, 0.77f, 0.454f, 0.402f, 0.679f, 0.555f, 0.453f, 0.426f, 0.275f, 0.626f, 0.78f, 0.454f});
	sInput.push_back({0.274f, 0.281f, 0.433f, 0.536f, 0.163f, 0.559f, 0.487f, 0.453f, 0.297f, 0.126f, 0.309f, 0.736f, 0.071f});
	sInput.push_back({0.713f, 0.184f, 0.476f, 0.299f, 0.522f, 0.559f, 0.54f, 0.151f, 0.382f, 0.39f, 0.358f, 0.707f, 0.558f});
	sInput.push_back({0.861f, 0.233f, 0.727f, 0.485f, 0.543f, 0.628f, 0.591f, 0.377f, 0.492f, 0.42f, 0.48f, 0.505f, 0.715f});
	sInput.push_back({0.353f, 0.077f, 0.428f, 0.433f, 0.185f, 0.869f, 0.582f, 0.113f, 0.461f, 0.27f, 0.602f, 0.586f, 0.101f});
	sInput.push_back({0.647f, 0.563f, 0.444f, 0.459f, 0.196f, 0.221f, 0.03f, 0.849f, 0.148f, 0.377f, 0.268f, 0.201f, 0.215f});
	sInput.push_back({0.392f, 0.334f, 0.433f, 0.536f, 0.196f, 0.541f, 0.407f, 0.245f, 0.256f, 0.061f, 0.341f, 0.553f, 0.034f});
	sInput.push_back({0.437f, 0.156f, 0.481f, 0.521f, 0.109f, 0.138f, 0.236f, 0.849f, 0.382f, 0.151f, 0.39f, 0.289f, 0.155f});
	sInput.push_back({0.645f, 0.184f, 0.684f, 0.613f, 0.207f, 0.559f, 0.16f, 0.736f, 0.593f, 0.893f, 0.073f, 0.187f, 0.244f});
	sInput.push_back({0.561f, 0.32f, 0.701f, 0.412f, 0.337f, 0.628f, 0.612f, 0.321f, 0.757f, 0.375f, 0.447f, 0.696f, 0.647f});
	sInput.push_back({0.979f, 0.196f, 0.551f, 0.041f, 0.228f, 0.731f, 0.707f, 0.566f, 0.757f, 0.352f, 0.626f, 0.535f, 0.622f});
	sInput.push_back({0.718f, 0.156f, 0.717f, 0.459f, 0.674f, 0.679f, 0.506f, 0.698f, 0.297f, 0.352f, 0.626f, 0.634f, 0.683f});
	sInput.push_back({0.737f, 0.18f, 0.663f, 0.34f, 0.261f, 0.507f, 0.559f, 0.17f, 0.593f, 0.369f, 0.618f, 0.769f, 0.704f});
	sInput.push_back({0.208f, 0.144f, 0.337f, 0.526f, 0.174f, 0.345f, 0.266f, 0.321f, 0.353f, 0.057f, 0.382f, 0.755f, 0.155f});
	sInput.push_back({0.532f, 0.259f, 0.995f, 0.742f, 0.587f, 0.569f, 0.494f, 0.642f, 0.476f, 0.196f, 0.528f, 0.707f, 0.394f});
	sInput.push_back({0.413f, 0.119f, 0.289f, 0.407f, 0.196f, 0.162f, 0.215f, 0.302f, 0.297f, 0.1f, 0.455f, 0.549f, 0.203f});
	sInput.push_back({0.711f, 0.715f, 0.481f, 0.613f, 0.196f, 0.103f, 0.027f, 0.736f, 0.233f, 0.456f, 0.244f, 0.176f, 0.173f});
#pragma endregion

	vector<vector<float>> sOutput;
#pragma region OutputData
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({1.f, 0.f, 0.f});
	sOutput.push_back({0.f, 1.f, 0.f});
	sOutput.push_back({0.f, 0.f, 1.f});
#pragma endregion

	vector<vector<float>> sTestInput;
#pragma region TestInputData
	sTestInput.push_back({0.7f, 0.498f, 0.631f, 0.485f, 0.402f, 0.293f, 0.046f, 0.698f, 0.123f, 0.392f, 0.39f, 0.201f, 0.287f});
	sTestInput.push_back({0.205f, 0.273f, 0.738f, 0.562f, 0.696f, 0.214f, 0.137f, 0.019f, 0.363f, 0.104f, 0.382f, 0.363f, 0.248f});
	sTestInput.push_back({0.255f, 0.532f, 0.342f, 0.433f, 0.185f, 0.352f, 0.274f, 0.453f, 0.461f, 0.f, 0.366f, 0.652f, 0.204f});
	sTestInput.push_back({0.539f, 0.625f, 0.535f, 0.562f, 0.467f, 0.148f, 0.222f, 0.396f, 0.23f, 0.693f, 0.073f, 0.022f, 0.194f});
	sTestInput.push_back({0.547f, 0.053f, 0.182f, 0.227f, 0.087f, 0.69f, 0.599f, 0.245f, 0.59f, 0.343f, 0.52f, 0.7f, 0.16f});
	sTestInput.push_back({0.695f, 0.101f, 0.299f, 0.381f, 0.261f, 0.386f, 0.306f, 0.358f, 0.101f, 0.215f, 0.61f, 0.436f, 0.251f});
	sTestInput.push_back({0.479f, 0.5f, 0.652f, 0.588f, 0.391f, 0.231f, 0.055f, 0.887f, 0.174f, 0.367f, 0.317f, 0.308f, 0.208f});
	sTestInput.push_back({0.366f, 0.729f, 0.733f, 0.82f, 0.348f, 0.421f, 0.378f, 0.566f, 0.41f, 0.068f, 0.358f, 0.678f, 0.062f});
#pragma endregion

	vector<vector<float>> sTestOutput;
#pragma region TestOutputData
	sTestOutput.push_back({0.f, 0.f, 1.f});
	sTestOutput.push_back({0.f, 1.f, 0.f});
	sTestOutput.push_back({0.f, 1.f, 0.f});
	sTestOutput.push_back({0.f, 0.f, 1.f});
	sTestOutput.push_back({0.f, 1.f, 0.f});
	sTestOutput.push_back({0.f, 1.f, 0.f});
	sTestOutput.push_back({0.f, 0.f, 1.f});
	sTestOutput.push_back({0.f, 1.f, 0.f});
#pragma endregion

	//Input		: 13
	//Hidden	: 5
	//Output	: 3
	MLP sNetwork;
	sNetwork.addLayerAtLast(L"input", 13u);
	sNetwork.addLayerAtLast(L"hidden", 100u);
	sNetwork.addLayerAtLast(L"output", 3u);

	sNetwork.getInputLayer()->resetLayerAll(-0.5f, 0.5f);
	//sNetwork.getInputLayer()->resetLayerAll(1.22f);
	const auto &sOutputVector = sNetwork.getOutputLayer()->getOutputVector();

	for(;;)
	{
		Layer::learnToBackAll(*sNetwork.getInputLayer(), *sNetwork.getOutputLayer(), 0.01f, 0.001f, 1000u, sInput, sOutput);

		uint32_t nIndex = 0u;
		for(auto &sInputVector : sTestInput)
		{
			++nIndex;
			sNetwork.getInputLayer()->calcOutputAll(sInputVector);
			wprintf_s(L"#%u : %.1f, %.1f, %.1f\n", nIndex, sOutputVector[0], sOutputVector[1], sOutputVector[2]);
		}

		wprintf_s(L"\n\n");
	}

	system("pause");

	//printf_s("Writing to file...");

	//std::ofstream sFileOutput{L"wine.pn", ofstream::binary | ofstream::out};
	//sNetwork.writeToFile(sFileOutput);

	//sFileOutput.flush();
	//sFileOutput.close();

	//printf_s(" Done.\n");
	//system("pause");

	return 0;
}

#endif