CC = nvcc
CFLAGS =
LIBS =
TARGETS = rectify pool convolve grid_4_4 grid_512_512

default: clean $(TARGETS)
all: default

.PHONY: clean

convolve:
	$(CC) -o convolve convolve.cu lodepng.cu $(CFLAGS) $(LIBS)

rectify:
	$(CC) -o rectify rectify.cu lodepng.cu $(CFLAGS) $(LIBS)

pool:
	$(CC) -o pool pool.cu lodepng.cu $(CFLAGS) $(LIBS)

grid_4_4:
	$(CC) grid.cu -o grid_4_4 $(CFLAGS) $(LIBS) -D RUNTIME_SMALL

grid_512_512:
	$(CC) grid.cu -o grid_512_512 $(CFLAGS) $(LIBS)

clean:
	-rm -f $(TARGETS)
	-rm -f *.o
