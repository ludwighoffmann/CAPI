This is the code necessary to run and analyze the main simulations of arXiv:2308.09992. Under CC0 license.

Python 3 code, tested on Mac and Linux with Python 3.11. Modifying the number of particles as well as the opening angle in the main simulation file allows to simulate the different densities and shapes described in the paper. Running the simulation code will produce csv files which contain the coordiantes of each particle over time. This can be analyzed to find the clustering dynamics. The conversion from particles to densities is described in the paper.
To test the code set the number of iterations to 1, the simulation will run ~1 hour.
The TDC code contains a slightly modified version of the main code where the density is increasing over time according to a specified function, as described in the SI.
