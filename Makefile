TARGET = rectify 
#change 'rectify' above for 'pool' or 'convolve' to compile the latter
CC = gcc
CFLAGS = -g -Wall
LIBS = -lpthread

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = rectify.c lodepng.c
#change 'rectify.c' above for 'pool.c' or 'convolve.c' to compile the latter
HEADERS = lodepng.h wm.h

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -std=c99 -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)