# Toy GIMPS in Rust
# This section is just for the Rust class grader:
## __What was built__:
Please read the "Real README" section below.

## __What Worked__:
This went pretty well! Not being able to use globals meant I was passing around more variables then I might've preferred, and clippy had some opinions about lots of stuff, but it really wasn't so bad overall. The program as a whole ended up a fair bit faster than I expected, though it's still miles away from real GIMPS programs (which was entirely expected). Rust's optimizations really impressed me; the difference between cargo run and cargo run --release is surprising. My time schedule for this also worked out pretty well; I was able to fit in more features than I'd originally intended. The end result was good enough quality that I decided to make it a sort of demonstration of GIMPS concepts that I'm gonna host publicly and share on the GIMPS forums.

## __What Didn't Work__:
All of my biggest troubles add up to looking for problems in the wrong places; I was too hasty to look for algorithmic problems rather than input problems (i.e. typos). Wasted a fair few hours debugging functions that weren't actually buggy this way. The program itself runs pretty great now, although the automatic signal length determination doesn't quite work as intended. If I had more time, I'd explore exactly why the formula doesn't work at larger exponents. I also didn't make enough tests, just in general; would've been good to have better unit tests throughout.

## __Lessons Learned__:
Mainly two things:
1. When the program doesn't work right, check thoroughly for input errors first before checking for algorithm errors.
2. Make tests as soon as I know what the structure of a function looks like (i.e. its arguments and return type).

But also some programming and a bunch of Rust-specific stuff (like the zip function on iters).

# Real README:
## Usage
To run, use
```
cargo run --release -- -e <exponent>
```
where "\<exponent\>" is the exponent you're trying to test. Run
```
cargo run -- -h
```
for a full list of options. Note that you'll need to pass in a signal length at higher exponents because the automatic signal length determination doesn't work quite right:
```
cargo run --release -- -e <exponent> -s <signal_length>
```
Though the signal length reported when you run it without the -s option gives you a good idea of where to start.

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
I make no guarantee that this program is the "real" way that _any_ other GIMPS program does _anything_; it's merely my attempt at illustrating some important GIMPS ideas in a (hopefully) digestable format. This program is missing quite a few key features that a real GIMPS program would have: interaction with the primenet server, save states that get regularly updated during runtime, P-1 factoring before the PRP loop, and probably a bunch of other things I'm not even aware of. Just use this as a tool to help understand the algorithm.