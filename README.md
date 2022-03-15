# Toy GIMPS in Rust

## Usage
To run, use
```
cargo run --release -- -e <exponent>
```
where "\<exponent\>" is the exponent you're trying to test. Run
```
cargo run -- -h
```
for a full list of options.

## Background
See [here](https://www.mersenne.org/) for GIMPS basics. 

The basic probable prime algorithm is:
```
q <- exponent of the prime to be tested
possible_prime <- 2^q - 1
res <- 3
for i in range 0 to q:
	res = (res^2) mod possible_prime 
if res is 9:
	possible_prime is a probable prime
else:
	possible_prime is not a probable prime
```

but the GIMPS programs in actual use are much, much more complicated than that. This program is intended to be a basic and readable implementation of the more advanced GIMPS PRP programs, for illustration purposes. If you want to _actually_ contribute to the search for primes, you should definitely use one of the tried-and-true GIMPS programs. This program implements:

### Bigger Features
- __The Irrational-Base Discrete Weighted Transform (IBDWT)__: see Richard Crandall and Barry Fagin's paper [here](https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1185244-1/S0025-5718-1994-1185244-1.pdf) starting on (pdf) page 10 with an example on page 12, and Richard Crandall and Carl Pomerance's book [here](http://thales.doa.fmph.uniba.sk/macaj/skola/teoriapoli/primes.pdf) starting on (pdf) page 506 with an example on page 507.
- __Gerbicz Error Checking__: see the original post from Gerbicz himself about Proth numbers [here](https://www.mersenneforum.org/showthread.php?t=22510), and the easier explanation about how to make it work for Mersenne numbers [here](https://www.mersenneforum.org/showpost.php?p=465584&postcount=95).

### Smaller Features
- __Balanced Digit Representation__: see [here](https://mersenneforum.org/showthread.php?t=27012&page=4) for discussion about this on the forum.
- __Automatic Signal Length Calculation__: see Richard Crandall, Ernst Mayer, and Jason Papadopoulos' paper [here](https://www.mersenneforum.org/attachments/pdfs/F24.pdf) starting on (pdf) page 8 with the actual formula on page 10. (Warning: this feature doesn't yet seem to work right with the rest of the program.)

## Disclaimer
I make no guarantee that this program is the "real" way that _any_ other GIMPS program does _anything_; it's merely my attempt at illustrating some important GIMPS ideas in a (hopefully) digestable format.