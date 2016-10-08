CC = gcc
CFLAGS =
LIBS = -lpthread
TARGETS = rectify pool convolve

default: clean $(TARGETS)
all: default

.PHONY: clean

rectify:
	$(CC) -o rectify rectify.c lodepng.c $(CFLAGS) -std=c99 $(LIBS)

pool:
	$(CC) -o pool pool.c lodepng.c $(CFLAGS) -std=c99 $(LIBS)

convolve:
	$(CC) -o convolve convolve.c lodepng.c $(CFLAGS) -std=c99 $(LIBS)

clean:
	-rm -f $(TARGETS)
	-rm -f *.o
