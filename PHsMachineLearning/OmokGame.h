
/*
	2017.02.02
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_OMOK_OMOK_GAME_H

#define _CLASS_OMOK_OMOK_GAME_H

/*
	TODO : Place your include directives here.
*/
#include <utility>

namespace Omok
{
	class OmokGame
	{
	private:
		/*
			TODO : Place your field declarations here.
		*/
		
		
	public:
		OmokGame();
		OmokGame(const OmokGame &rSrc);
		OmokGame(OmokGame &&rSrc);
		~OmokGame();
		/*
			TODO : Place your other constructors here.
		*/
		
		
	public:
		OmokGame &operator=(const OmokGame &rSrc);
		OmokGame &operator=(OmokGame &&rSrc);
		/*
			TODO : Place your other operator overloadings here.
		*/
		
		
	public:
		/*
			TODO : Place your member function declarations here.
		*/
		
	};
}

#endif