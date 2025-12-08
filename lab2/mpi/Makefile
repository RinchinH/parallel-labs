CXX       = mpicxx
CXXFLAGS  = -O3 -march=native -std=c++17

all: jacobi_mpi

jacobi_mpi: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Профилирование (если есть mpecc)
jacobi_mpi_mpe: main.cpp
	mpecc -mpilog -cc=mpicxx -o $@ $<

clean:
	rm -f jacobi_mpi jacobi_mpi_mpe *.clog2 *.slog2
