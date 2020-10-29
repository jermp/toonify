## Toonify Images for Painting

A C++ program that turns an image into a toon that can be painted
using a color palette.

The algorithm is based on the paper [*Toonify: Cartoon Photo Effect Application*](https://stacks.stanford.edu/file/druid:yt916dh6570/Dade_Toonify.pdf).

### Install, Compile, and Run

After installation of `opencv` via

	brew install opencv
	
compile the C++ program `toonify.cpp` with

	make
	
Usage

	./toonify [path_to_file] [num_colors] [blur_level]
	