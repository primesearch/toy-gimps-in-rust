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
    let mut args_list = std::env::args();
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
    let (_bit_array, two_to_the_bit_array, weight_array, fft, ifft) =
        initialize_constants(exponent, signal_length);

    let mut residue = vec![0.0; signal_length];

    for i in 0..exponent {
        if i % update_frequency == 0 {
            println!("Iteration: {:?}", i);
            println!("Percent Done: {:.2}%", (i as f64 / exponent as f64) * 100.0);
            // println!("Current Roundoff Error: {:.4}", roundoff);
            println!("Residue: {:?}", &residue);
        }
        residue = squaremod_with_ibdwt(residue, &two_to_the_bit_array, &weight_array, &fft, &ifft);
    }

    todo!()
}

fn squaremod_with_ibdwt(
    signal: Vec<f64>,
    two_to_the_bit_array: &[f64],
    weight_array: &[f64],
    fft: &std::sync::Arc<dyn Fft<f64>>,
    ifft: &std::sync::Arc<dyn Fft<f64>>,
) -> Vec<f64> {
    let balanced_signal = balance(signal, two_to_the_bit_array);
    let transformed_signal = weighted_transform(balanced_signal, weight_array, fft);

    let squared_transformed_signal = transformed_signal.into_iter().map(|x| x * x).collect();

    let squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array, ifft);
    let rounded_signal = squared_signal.into_iter().map(|x| x.round()).collect();

    complete_carry(rounded_signal, two_to_the_bit_array)
}

fn balance(mut signal: Vec<f64>, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut carry_val = 0.0;
    let mut i = 0;
    while i < signal.len() {
        signal[i] += carry_val;
        if signal[i] > two_to_the_bit_array[i] / 2.0 {
            signal[i] -= two_to_the_bit_array[i];
            carry_val = 1.0;
        } else {
            carry_val = 0.0;
        }
        i += 1;
    }
    signal
}

fn weighted_transform(
    mut signal: Vec<f64>,
    weight_array: &[f64],
    fft: &std::sync::Arc<dyn Fft<f64>>,
) -> Vec<Complex<f64>> {
    let mut complex_signal = Vec::new();

    let mut i = 0;
    while i < signal.len() {
        signal[i] *= weight_array[i];
        complex_signal.push(Complex {
            re: signal[i],
            im: 0.0,
        });
        i += 1;
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

    let mut i = 0;
    while i < complex_signal.len() {
        real_signal.push(complex_signal[i].re);
        real_signal[i] /= weight_array[i];
        i += 1;
    }

    real_signal
}

fn complete_carry(mut signal: Vec<f64>, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut carry_val = 0.0;
    let signal_length = signal.len();
    let mut i = 0;
    while i < signal_length {
        signal[i] += carry_val;
        carry_val = (signal[i] / two_to_the_bit_array[i]).floor();
        signal[i] %= two_to_the_bit_array[i];
        i += 1;
    }
    while carry_val != 0.0 {
        signal[i % signal_length] += carry_val;
        carry_val = (signal[i % signal_length] / two_to_the_bit_array[i % signal_length]).floor();
        signal[i % signal_length] %= two_to_the_bit_array[i % signal_length];
        i += 1;
    }

    signal
}

fn initialize_constants(
    exponent: usize,
    signal_length: usize,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    sync::Arc<dyn Fft<f64>>,
    sync::Arc<dyn Fft<f64>>,
) {
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

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal_length);
    let ifft = planner.plan_fft_inverse(signal_length);

    (bit_array, two_to_the_bit_array, weight_array, fft, ifft)
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
