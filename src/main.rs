//! Determines if the Mersenne number corresponding to the passed in exponent is
//! a prime number by running a PRP test with the IBDWT. See readme for details.
//!
//! Josh McFerran, Winter 2022, Programming in Rust

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync;

/// Retreives the command line args and checks them for validity before passing the bulk
/// of the work to prptest
fn main() {
    let (mut exponent, mut signal_length, mut update_frequency) = (0, 0, 0);
    let mut run_verbose = false;
    let mut args_list = std::env::args();
    args_list.next();
    while let Some(argument) = args_list.next() {
        match argument.as_str() {
            "-e" | "-E" | "-exp" | "--exponent" => {
                if let Some(next_arg) = args_list.next() {
                    exponent = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-s" | "-S" | "-sig" | "--siglen" => {
                if let Some(next_arg) = args_list.next() {
                    signal_length = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-f" | "-F" | "-freq" | "--frequency" => {
                if let Some(next_arg) = args_list.next() {
                    update_frequency = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-v" | "-V" | "--verbose" => run_verbose = true,
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

    if prptest(exponent, signal_length, update_frequency, run_verbose) {
        println!("2 ^ {:?} - 1 is probably prime!", exponent);
    } else {
        println!("2 ^ {:?} - 1 is not prime.", exponent);
    }
}

/// This runs the actual loop and is the bulk of the program. Returns true if 2 ^ exponent -1
/// is probably prime, false otherwise.
fn prptest(
    exponent: usize,
    signal_length: usize,
    update_frequency: usize,
    run_verbose: bool,
) -> bool {
    let (bit_array, two_to_the_bit_array) = init_bit_array(exponent, signal_length);
    let weight_array = init_weight_array(exponent, signal_length);
    let (fft, ifft) = init_fft(signal_length);

    let mut residue = signalize(3, &two_to_the_bit_array);

    if run_verbose {
        println!(
            "Bit Array: {:?}\ntwo_to_the_bit_array: {:?}\n",
            bit_array, two_to_the_bit_array
        );
    }

    for i in 0..exponent {
        if i % update_frequency == 0 {
            println!("Iteration: {:?}", i);
            println!("{:.2}% Finished", (i as f64 / exponent as f64) * 100.0);
            // println!("Current Roundoff Error: {:.4}", roundoff);
            println!("Residue: {:?}\n", &residue);
        }
        if run_verbose {
            println!("Designalized: {:?}", designalize(&residue, &two_to_the_bit_array));
        }
        residue = squaremod_with_ibdwt(
            residue,
            &two_to_the_bit_array,
            &weight_array,
            &fft,
            &ifft,
            run_verbose,
        );
    }

    println!("Final Residue: {:?}", residue);

    designalize(&residue, &two_to_the_bit_array) == 9
}

fn squaremod_with_ibdwt(
    signal: Vec<f64>,
    two_to_the_bit_array: &[f64],
    weight_array: &[f64],
    fft: &std::sync::Arc<dyn Fft<f64>>,
    ifft: &std::sync::Arc<dyn Fft<f64>>,
    run_verbose: bool,
) -> Vec<f64> {
    let balanced_signal = balance(signal, two_to_the_bit_array);
    if run_verbose {
        println!("Balanced: {:?}", balanced_signal);
    }

    let transformed_signal = weighted_transform(balanced_signal, weight_array, fft);
    if run_verbose {
        println!("fft: {:?}", transformed_signal);
    }
    
    let squared_transformed_signal = transformed_signal.into_iter().map(|x| x * x).collect();
    if run_verbose {
        println!("Squared fft: {:?}", squared_transformed_signal);
    }
    
    let squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array, ifft);
    if run_verbose {
        println!("ifft:  {:?}", squared_signal);
    }

    let rounded_signal = squared_signal.into_iter().map(|x| x.round()).collect();
    if run_verbose {
        println!("Rounded:  {:?}\n\n", rounded_signal);
    }

    complete_carry(rounded_signal, two_to_the_bit_array)
}

fn balance(mut signal: Vec<f64>, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut carry_val = 0.0;
    for i in 0..signal.len() {
        signal[i] += carry_val;
        if signal[i] > two_to_the_bit_array[i] / 2.0 {
            signal[i] -= two_to_the_bit_array[i];
            carry_val = 1.0;
        } else {
            carry_val = 0.0;
        }
    }
    signal
}

fn weighted_transform(
    mut signal: Vec<f64>,
    weight_array: &[f64],
    fft: &std::sync::Arc<dyn Fft<f64>>,
) -> Vec<Complex<f64>> {
    let mut complex_signal = Vec::new();

    for i in 0..signal.len() {
        signal[i] *= weight_array[i];
        complex_signal.push(Complex {
            re: signal[i],
            im: 0.0,
        });
    }

    fft.process(&mut complex_signal);
    complex_signal
}

fn inverse_weighted_transform(
    mut complex_signal: Vec<Complex<f64>>,
    weight_array: &[f64],
    ifft: &std::sync::Arc<dyn Fft<f64>>,
) -> Vec<f64> {
    let mut real_signal = Vec::new();

    ifft.process(&mut complex_signal);
    println!("Unweighted ifft: {:?}", complex_signal);

    for i in 0..complex_signal.len() {
        real_signal.push(complex_signal[i].re);
        real_signal[i] /= weight_array[i];
    }

    real_signal
}

fn complete_carry(mut signal: Vec<f64>, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut carry_val = 0.0;
    let signal_length = signal.len();
    for i in 0..signal_length {
        signal[i] += carry_val;
        carry_val = (signal[i] / two_to_the_bit_array[i]).floor();
        signal[i] %= two_to_the_bit_array[i];
    }
    let mut i = 0;
    while carry_val != 0.0 {
        signal[i % signal_length] += carry_val;
        carry_val = (signal[i % signal_length] / two_to_the_bit_array[i % signal_length]).floor();
        signal[i % signal_length] %= two_to_the_bit_array[i % signal_length];
        i += 1;
    }

    signal
}

fn init_bit_array(exponent: usize, signal_length: usize) -> (Vec<f64>, Vec<f64>) {
    let fexponent: f64 = exponent as f64;
    let fsignal_length: f64 = signal_length as f64;
    let mut bit_array = Vec::new();

    let mut fi = 1.0;
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
    (bit_array, two_to_the_bit_array)
}

fn init_weight_array(exponent: usize, signal_length: usize) -> Vec<f64> {
    let fexponent: f64 = exponent as f64;
    let fsignal_length: f64 = signal_length as f64;
    let mut weight_array = Vec::new();

    let mut fi = 0.0;
    while fi < fsignal_length {
        weight_array.push(
            2_f64.powf((fexponent * fi) / fsignal_length).ceil()
                - ((fexponent * fi) / fsignal_length),
        );
        fi += 1.0;
    }
    weight_array
}

fn init_fft(signal_length: usize) -> (sync::Arc<dyn Fft<f64>>, sync::Arc<dyn Fft<f64>>) {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal_length);
    let ifft = planner.plan_fft_inverse(signal_length);

    (fft, ifft)
}

fn signalize(mut num_to_signalize: usize, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut signal = vec![0.0_f64; two_to_the_bit_array.len()];
    let mut i = 0;
    while num_to_signalize > 0 {
        signal[i] = num_to_signalize as f64 % two_to_the_bit_array[i];
        num_to_signalize /= two_to_the_bit_array[i] as usize;
        i += 1;
    }
    signal
}

fn designalize(signal: &[f64], two_to_the_bit_array: &[f64]) -> i64 {
    let mut base = 1;
    let mut resultant_num = 0;
    for i in 0..signal.len() {
        resultant_num += signal[i] as i64 * base;
        base *= two_to_the_bit_array[i] as i64;
    }
    resultant_num
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
