#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 2
#define BYTES_PER_PIXEL 4

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

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

unsigned get_block_offset(unsigned k, unsigned image_width){
	return BYTES_PER_PIXEL * BLOCK_SIZE * ( image_width * (k / (image_width / 2)) +  (k % (image_width / 2)));
}

int main(void){
	int a = 8, b = 17;
	int kek = pool_offset(a,b);
	int kekek = get_block_offset(a,b);
	printf("kek %d, kekek %d; for block %d, width %d\n",kek,kekek,a,b);
}

