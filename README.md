# Toy GIMPS in Rust

See [here](https://www.mersenne.org/) for GIMPS basics. In my capstone, I’m working on an open source GIMPS program ((([TensorPrime](https://github.com/TPU-Mersenne-Prime-Search/TensorPrime)) to run on the TPUs google provides through colab. That program is in python and includes some specific details to make it work on the TPU, so I’d like to recreate it (or most of it, anyways) in Rust as a command-line program that runs on the CPU. Basic usage will be to run it on the command line and pass it the exponent of the Mersenne number you’d like to test; then, after it finishes testing, it outputs whether it’s a probable prime or not. Since I’ve already researched the details of the irrational base discrete weighted transform (IBDWT) and Gerbicz error checking (GEC)  I’d like those to be implemented as well as the basic probable prime algorithm. I don’t plan to implement the more complex optimizations that pre-existing GIMPS programs use, so I don’t expect my project to be competitive with them speed-wise, but I’d at least like my program to be correct.

The basic probable prime algorithm is:
	q <- exponent of the prime to be tested
	possible_prime <- 2^q - 1
res <- 3
for i in range 0 to q:
	res = (res^2) mod possible_prime 
if res is 9:
	possible_prime is a probable prime
	else:
		possible_prime is not a probable prime

And that’s it. The idea is to use the IBDWT to speed up the squaring process (since the numbers we’re squaring here are heckin’ big), and to use GEC to check every 100,000 iterations or so if we’ve made an error and rollback to a previously saved state if so. I don’t plan on having much in the way of logging information or different run modes like the other GIMPS programs have (since I want this program to be a few hundred lines, not several thousand); probably the only optional flags I’ll include in the final program are a way to time the total process length, a way to give some sort of progress report so you know how many more iterations there are to do, and a way to manually determine some of the internal details of the program (in particular, the IBDWT signal length).
