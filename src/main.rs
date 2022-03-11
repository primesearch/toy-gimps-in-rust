//! Determines if the Mersenne number corresponding to the passed in exponent is
//! a prime number by running a PRP test with the IBDWT. See readme for details.
//!
//! Josh McFerran, Winter 2022, Programming in Rust

use std::env;

/// Retreives the command line args and checks them for validity before passing the bulk
/// of the work to prptest
fn main() {
    let (mut exponent, mut signal_length, mut update_frequency) = (0, 0, 0);
    let mut args_list = env::args();
    while let Some(argument) = args_list.next() {
        match argument.as_str() {
            "-e" | "-exp" | "--exponent" => {
                if let Some(next_arg) = args_list.next() {
                    exponent = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-s" | "-sig" | "--siglen" => {
                if let Some(next_arg) = args_list.next() {
                    signal_length = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-f" | "-freq" | "--frequency" => {
                if let Some(next_arg) = args_list.next() {
                    update_frequency = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            &_ => usage(),
        }
    }

    assert!(exponent > 3, "Invalid exponent; must be greater than 3");
    assert!(
        signal_length < exponent,
        "Invalid signal length; must be less than the exponent"
    );

    if update_frequency == 0 {
        update_frequency = ((exponent as f64) / 100.0).ceil() as usize;
    }

    if prptest(exponent, signal_length, update_frequency) {
        println!("2 ^ {:?} - 1 is probably prime!", exponent);
    } else {
        println!("2 ^ {:?} - 1 is not prime.", exponent);
    }
}

/// This runs the actual loop and is the bulk of the program. Returns true if 2 ^ exponent -1
/// is probably prime, false otherwise.
fn prptest(exponent: usize, signal_length: usize, update_frequency: usize) -> bool {
    let (bit_array, two_to_the_bit_array, weight_array) =
        initialize_constants(exponent, signal_length);

    let mut residue = vec![0.0; signal_length];

    for i in 0..exponent {
        if i % update_frequency == 0 {
            println!("Iteration: {:?}", i);
            println!("Percent Done: {:.2}%", (i as f64 / exponent as f64) * 100.0);
            // println!("Current Roundoff Error: {:.4}", roundoff);
            println!("Residue: {:?}", &residue);
        }
        residue = squaremod_with_ibdwt(residue, &two_to_the_bit_array, &weight_array);
    }

    todo!()
}

fn squaremod_with_ibdwt(
    signal: Vec<f64>,
    two_to_the_bit_array: &Vec<f64>,
    weight_array: &Vec<f64>,
) -> Vec<f64> {
    let balanced_signal = balance(signal, two_to_the_bit_array);
    let transformed_signal = weighted_transform(balanced_signal, weight_array);

    let squared_transformed_signal = transformed_signal.into_iter().map(|x| x * x).collect();

    let squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array);
    let rounded_signal = squared_signal.into_iter().map(|x| x.round()).collect();
    let carried_signal = partial_carry(rounded_signal, two_to_the_bit_array);

    carried_signal
}

fn balance(signal: Vec<f64>, two_to_the_bit_array: &Vec<f64>) -> Vec<f64> {
    todo!()
}

fn weighted_transform(signal: Vec<f64>, weight_array: &Vec<f64>) -> Vec<f64> {
    todo!()
}

fn inverse_weighted_transform(signal: Vec<f64>, weight_array: &Vec<f64>) -> Vec<f64> {
    todo!()
}

fn partial_carry(signal: Vec<f64>, two_to_the_bit_array: &Vec<f64>) -> Vec<f64> {
    todo!()
}

fn initialize_constants(exponent: usize, signal_length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fexponent: f64 = exponent as f64;
    let fsignal_length: f64 = signal_length as f64;
    let mut fi: f64;

    let mut bit_array = Vec::new();
    fi = 1.0;
    while fi < fsignal_length + 1.0 {
        bit_array.push(
            ((fexponent * fi) / fsignal_length).ceil()
                - ((fexponent * (fi - 1.0)) / fsignal_length).ceil(),
        );
        fi += 1.0;
    }

    let mut two_to_the_bit_array = Vec::new();
    for i in bit_array.iter() {
        two_to_the_bit_array.push(2_f64.powf(*i));
    }

    let mut weight_array = Vec::new();
    fi = 0.0;
    while fi < fsignal_length {
        weight_array.push(
            2_f64.powf((fexponent * fi) / fsignal_length).ceil()
                - ((fexponent * fi) / fsignal_length),
        );
        fi += 1.0;
    }

    (bit_array, two_to_the_bit_array, weight_array)
}

/// Parse the given string as a `u128`.
fn parsenum(s: String) -> usize {
    s.parse().unwrap_or_else(|_| usage())
}

/// Print a usage error message and exit.
fn usage() -> ! {
    eprintln!("usage: toy-gimps-in-rust -e exponent -s signal_length [-f update_frequency]");
    std::process::exit(1);
}
