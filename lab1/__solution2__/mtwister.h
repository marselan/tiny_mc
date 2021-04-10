#ifndef __MTWISTER_H
#define __MTWISTER_H

#define STATE_VECTOR_LENGTH 624
#define STATE_VECTOR_M      397 /* changes to STATE_VECTOR_LENGTH also require changes to this */

typedef struct tagMTRand {			// struct define una estructura
  unsigned long mt[STATE_VECTOR_LENGTH];	// unsigened long define el 1er miembro de la estructura
  int index;					// int define el 2do miembro de la estructura
} MTRand;					// typedef crea un sinónimo de tagMTRand denominado MTRand

MTRand seedRand(unsigned long seed);
unsigned long genRandLong(MTRand* rand);
double genRand(MTRand* rand);

#endif /* #ifndef __MTWISTER_H */
