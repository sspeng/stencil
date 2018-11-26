
CC = mpiicc 
OPT = -Ofast
CFLAGS = -Wall -std=c99 $(OPT) -qopenmp -xavx2
LDFLAGS = -Wall -qopenmp
LDLIBS = $(LDFLAGS)

targets = benchmark-naive benchmark-optimized benchmark-openmp-naive \
			benchmark-openmp-oblivious benchmark-openmp-timeskew benchmark-openmp-circqueue \
			benchmark-mpi-naive benchmark-mpi-3d
objects = check.o benchmark.o stencil-naive.o stencil-optimized.o stencil-openmp-naive.o \
			stencil-openmp-oblivious.o stencil-openmp-timeskew.o stencil-openmp-circqueue.o \
			stencil-mpi-naive.o stencil-mpi-3d.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : check.o benchmark.o stencil-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-optimized : check.o benchmark.o stencil-optimized.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-openmp-naive : check.o benchmark.o stencil-openmp-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-openmp-oblivious : check.o benchmark.o stencil-openmp-oblivious.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-openmp-timeskew : check.o benchmark.o stencil-openmp-timeskew.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-openmp-circqueue : check.o benchmark.o stencil-openmp-circqueue.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-mpi-naive : check.o benchmark.o stencil-mpi-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-mpi-3d : check.o benchmark.o stencil-mpi-3d.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)