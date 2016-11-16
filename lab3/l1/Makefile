CC = nvcc
CFLAGS =
LIBS =
TARGETS = rectify pool convolve

default: clean $(TARGETS)
all: default

.PHONY: clean

convolve:
	$(CC) -o convolve convolve.cu lodepng.cu $(CFLAGS) $(LIBS)

rectify:
	$(CC) -o rectify rectify.cu lodepng.cu $(CFLAGS) $(LIBS)

pool:
	$(CC) -o pool pool.cu lodepng.cu $(CFLAGS) $(LIBS)

clean:
	-rm -f $(TARGETS)
	-rm -f *.o
