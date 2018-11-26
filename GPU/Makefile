
CC = nvcc 
OPT = 
CFLAGS = $(OPT)
LDFLAGS =
LDLIBS = $(LDFLAGS)

targets = benchmark-naive benchmark-optimized benchmark-oblivious \
		  benchmark-timeskew benchmark-circqueue
objects = check.o benchmark.o stencil-naive.o stencil-optimized.o stencil-oblivious.o \
	      stencil-timeskew.o stencil-circqueue.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : check.o benchmark.o stencil-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-optimized : check.o benchmark.o stencil-optimized.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-oblivious : check.o benchmark.o stencil-oblivious.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-timeskew : check.o benchmark.o stencil-timeskew.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-circqueue : check.o benchmark.o stencil-circqueue.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.cu common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)