all: en1303_macs_differs

en1303_macs_differs: en1303_macs_differs.cpp
	${CXX} en1303_macs_differs.cpp -o en1303_macs_differs -O3 -march=native -fopenmp -lgomp 
