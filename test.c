#include <stdio.h>
#include <stdlib.h>

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

unsigned get_block_offset(unsigned i, unsigned j, unsigned image_width)
{
	return ((i + 1) * image_width + j + 1);
}

unsigned pool_offset(unsigned k, unsigned image_width){
	image_width = image_width - image_width % 2;
	return 4 * (( 2 * image_width * (k / (image_width / 2))) + 2 * (k % (image_width / 2)));
}

/**
* TODO comment this
*/
unsigned convert_block_to_pixel_offset(unsigned blocks_offset, unsigned image_width)
{
	int i = get_2d(blocks_offset, image_width - 2, 0);
	int j = get_2d(blocks_offset, image_width - 2, 1);
	printf("i%d, j%d",i,j);
	return get_1d(i,j,image_width);
}

int main(void){
	int kek = pool_offset(8,17);
	printf("kek %d\n",kek);
}