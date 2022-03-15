// SPDX-FileCopyrightText: © 2022 Joshua McFerran <mcferran.joshua@gmail.com>
// SPDX-License-Identifier: MIT


//! Determines if the Mersenne number corresponding to the passed in exponent is
//! a prime number by running a PRP test with the IBDWT. See readme for details.
//!
//! Josh McFerran, Winter 2022, Programming in Rust
// use realfft::RealFftPlanner;
// use rustfft::{num_complex::Complex, FftPlanner};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

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
            "-h" | "-H" | "--help" => usage_help(),
            &_ => usage(),
        }
    }

    if signal_length == 0 {
        signal_length = determine_best_signal_length(exponent);
        println!("Taking signal length {:?}", signal_length);
    }

    assert!(exponent > 1, "Invalid exponent; must be greater than 1");
    assert!(
        signal_length > 1,
        "Invalid signal length; must be greater than 1"
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

/// This runs the actual loop and is the bulk of the program. Returns true if 2 ^ exponent - 1
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
    let mut roundoff = 0.0;
    let mut max_roundoff = 0.0;

    let mut residue = vec![0.0; signal_length];
    residue[0] = 3.0;
    residue = complete_carry(residue, &two_to_the_bit_array);

    let gec_l: usize = 2000;
    let gec_l_2 = gec_l.pow(2);
    let mut gec_d = residue.clone();
    let mut gec_prev_d = residue.clone();
    let mut gec_saved_d = residue.clone();
    let mut gec_saved_residue = residue.clone();
    let mut gec_saved_i: usize = 0;
    let three_signal = residue.clone();

    if signal_length <= 5 {
        println!(
            "Bit Array: {:?}\ntwo_to_the_bit_array: {:?}\nWeight Array:{:?}\n",
            bit_array, two_to_the_bit_array, weight_array
        );
    } else {
        println!(
            "Bit Array: [{:?}, {:?}, {:?}, {:?}, {:?} ...]",
            bit_array[0], bit_array[1], bit_array[2], bit_array[3], bit_array[4]
        );
        println!(
            "two_to_the_bit_array: [{:?}, {:?}, {:?}, {:?}, {:?} ...]",
            two_to_the_bit_array[0],
            two_to_the_bit_array[1],
            two_to_the_bit_array[2],
            two_to_the_bit_array[3],
            two_to_the_bit_array[4]
        );
        println!(
            "Weight Array: [{:?}, {:?}, {:?}, {:?}, {:?} ...]\n",
            weight_array[0], weight_array[1], weight_array[2], weight_array[3], weight_array[4]
        );
    }

    let mut i = 0;
    while i < exponent {
        if i % update_frequency == 0 {
            println!("Iteration: {:?}", i);
            println!("{:.2}% Finished", (i as f64 / exponent as f64) * 100.0);
            println!("Current Roundoff Error: {:.4}", roundoff);
            if signal_length <= 5 {
                println!("Residue: {:?}\n", &residue);
            } else {
                println!(
                    "Residue: [{:?}, {:?}, {:?}, {:?}, {:?}, ...]\n",
                    residue[0], residue[1], residue[2], residue[3], residue[4]
                );
            }
        }
        if i % gec_l == 0 && i != 0 {
            gec_prev_d = gec_d.clone();
            let (temp_gec_d, temp_roundoff) = multmod_with_ibdwt(
                residue.clone(),
                gec_d,
                &two_to_the_bit_array,
                &weight_array,
                &fft,
                &ifft,
                false,
            );
            gec_d = temp_gec_d;
            check_roundoff(&temp_roundoff, &i);
        }
        if (i % gec_l_2 == 0 && i != 0) || (i % gec_l == 0 && i + gec_l >= exponent) {
            let mut check_value = gec_prev_d.clone();
            for _j in 0..gec_l {
                let (temp_check_value, temp_roundoff) = squaremod_with_ibdwt(
                    check_value,
                    &two_to_the_bit_array,
                    &weight_array,
                    &fft,
                    &ifft,
                    false,
                );
                check_roundoff(&temp_roundoff, &i);
                check_value = temp_check_value;
            }
            let (temp_check_value, temp_roundoff) = multmod_with_ibdwt(
                check_value,
                three_signal.clone(),
                &two_to_the_bit_array,
                &weight_array,
                &fft,
                &ifft,
                false,
            );
            check_roundoff(&temp_roundoff, &i);
            if temp_check_value != gec_d {
                println!(
                    "Hardware error detected between iterations {:?} and {:?}; reverting to {:?}",
                    gec_saved_i, i, gec_saved_i
                );
                gec_d = gec_saved_d.clone();
                residue = gec_saved_residue.clone();
                i = gec_saved_i;
            } else {
                gec_saved_d = gec_d.clone();
                gec_saved_residue = residue.clone();
                gec_saved_i = i;
            }
        }
        let (temp_residue, temp_roundoff) = squaremod_with_ibdwt(
            residue,
            &two_to_the_bit_array,
            &weight_array,
            &fft,
            &ifft,
            run_verbose,
        );
        residue = temp_residue;
        roundoff = check_roundoff(&temp_roundoff, &i);
        max_roundoff = roundoff.max(max_roundoff);

        i += 1;
    }

    if signal_length <= 5 {
        println!("\nFinal residue: {:?}", &residue);
    } else {
        println!(
            "\nFinal residue: [{:?}, {:?}, {:?}, {:?}, {:?}, ...]",
            residue[0], residue[1], residue[2], residue[3], residue[4]
        );
    }
    println!("Max Roundoff Error: {:.4}", max_roundoff);

    let mut nine_signal = vec![0.0; signal_length];
    nine_signal[0] = 9.0;
    nine_signal = complete_carry(nine_signal, &two_to_the_bit_array);

    residue == nine_signal
}

fn squaremod_with_ibdwt(
    signal: Vec<f64>,
    two_to_the_bit_array: &[f64],
    weight_array: &[f64],
    fft: &Arc<dyn RealToComplex<f64>>,
    ifft: &Arc<dyn ComplexToReal<f64>>,
    run_verbose: bool,
) -> (Vec<f64>, f64) {
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
        println!("ifft: {:?}", squared_signal);
    }

    let rounded_signal: Vec<f64> = squared_signal.iter().map(|&x| x.round()).collect();
    if run_verbose {
        println!("Rounded: {:?}", rounded_signal);
    }
    let roundoff = squared_signal
        .iter()
        .zip(rounded_signal.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max);

    let carried_signal = complete_carry(rounded_signal, two_to_the_bit_array);
    if run_verbose {
        println!("Carried: {:?}\n\n", carried_signal);
    }

    (carried_signal, roundoff)
}

fn multmod_with_ibdwt(
    first_signal: Vec<f64>,
    second_signal: Vec<f64>,
    two_to_the_bit_array: &[f64],
    weight_array: &[f64],
    fft: &Arc<dyn RealToComplex<f64>>,
    ifft: &Arc<dyn ComplexToReal<f64>>,
    run_verbose: bool,
) -> (Vec<f64>, f64) {
    let first_balanced_signal = balance(first_signal, two_to_the_bit_array);
    if run_verbose {
        println!("First Balanced: {:?}", first_balanced_signal);
    }

    let first_transformed_signal = weighted_transform(first_balanced_signal, weight_array, fft);
    if run_verbose {
        println!("fft: {:?}", first_transformed_signal);
    }

    let second_balanced_signal = balance(second_signal, two_to_the_bit_array);
    if run_verbose {
        println!("Second Balanced: {:?}", second_balanced_signal);
    }

    let second_transformed_signal = weighted_transform(second_balanced_signal, weight_array, fft);
    if run_verbose {
        println!("fft: {:?}", second_transformed_signal);
    }

    let multiplied_transformed_signal = first_transformed_signal
        .into_iter()
        .zip(second_transformed_signal.into_iter())
        .map(|(x, y)| x * y)
        .collect();
    if run_verbose {
        println!("Squared fft: {:?}", multiplied_transformed_signal);
    }

    let squared_signal =
        inverse_weighted_transform(multiplied_transformed_signal, weight_array, ifft);
    if run_verbose {
        println!("ifft: {:?}", squared_signal);
    }

    let rounded_signal: Vec<f64> = squared_signal.iter().map(|&x| x.round()).collect();
    if run_verbose {
        println!("Rounded: {:?}", rounded_signal);
    }
    let roundoff = squared_signal
        .iter()
        .zip(rounded_signal.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max);

    let carried_signal = complete_carry(rounded_signal, two_to_the_bit_array);
    if run_verbose {
        println!("Carried: {:?}\n\n", carried_signal);
    }

    (carried_signal, roundoff)
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
    signal[0] += carry_val;
    signal
}

fn weighted_transform(
    signal: Vec<f64>,
    weight_array: &[f64],
    fft: &Arc<dyn RealToComplex<f64>>,
) -> Vec<Complex<f64>> {
    let mut complex_signal = fft.make_output_vec();
    let mut weighted_signal = Vec::new();

    for i in 0..signal.len() {
        weighted_signal.push(signal[i] * weight_array[i]);
    }
    fft.process(&mut weighted_signal, &mut complex_signal)
        .unwrap();

    complex_signal
}

fn inverse_weighted_transform(
    mut complex_signal: Vec<Complex<f64>>,
    weight_array: &[f64],
    ifft: &Arc<dyn ComplexToReal<f64>>,
) -> Vec<f64> {
    let mut real_signal = ifft.make_output_vec();
    let signal_length = real_signal.len() as f64;
    ifft.process(&mut complex_signal, &mut real_signal).unwrap();
    for i in 0..real_signal.len() {
        real_signal[i] /= weight_array[i] * signal_length;
    }
    real_signal
}

fn complete_carry(mut signal: Vec<f64>, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut carry_val = 0.0;
    let signal_length = signal.len();
    for i in 0..signal_length {
        signal[i] += carry_val;
        carry_val = (signal[i] / two_to_the_bit_array[i]).floor();
        signal[i] = ((signal[i] % two_to_the_bit_array[i]) + two_to_the_bit_array[i])
            % two_to_the_bit_array[i];
    }
    let mut i = 0;
    let mut index;
    while carry_val != 0.0 {
        index = i % signal_length;
        signal[index] += carry_val;
        carry_val = (signal[index] / two_to_the_bit_array[index]).floor();
        signal[index] = ((signal[index] % two_to_the_bit_array[index])
            + two_to_the_bit_array[index])
            % two_to_the_bit_array[index];
        i += 1;
    }

    signal
}

fn determine_best_signal_length(exponent: usize) -> usize {
    let mut best_signal_length = 0;
    let mut base = 0;
    while best_signal_length == 0 {
        for i in 128..255 {
            let prospective_signal_length = i * (1 << base);
            let max = get_max_exponent(prospective_signal_length as f64);
            if max > exponent {
                best_signal_length = prospective_signal_length;
                break;
            }
        }
        base += 1;
    }
    best_signal_length
}

fn get_max_exponent(signal_length: f64) -> usize {
    let num_mantissa_bits = 53.0;
    let magic_c = 14.0;
    let ln_2_inverse = 1.0 / 2.0_f64.ln();

    let ln_signal_length = signal_length.ln();
    let lnln_signal_length = ln_signal_length.ln();
    let log2_signal_length = ln_2_inverse * ln_signal_length;
    let lnlog2_signal_length = log2_signal_length.ln();
    let log2log2_signal_length = ln_2_inverse * lnlog2_signal_length;
    let lnlnln_signal_length = lnln_signal_length.ln();
    let log2lnln_signal_length = ln_2_inverse * lnlnln_signal_length;

    let max_bit_array_value = 0.5
        * (num_mantissa_bits
            - magic_c
            - 0.5 * (log2_signal_length + log2log2_signal_length)
            - 1.5 * (log2lnln_signal_length));
    (max_bit_array_value * signal_length) as usize
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

    assert!(
        bit_array.iter().all(|&x| x < 30.0),
        "Signal length too small for this exponent."
    );
    let mut two_to_the_bit_array = Vec::new();
    for &i in bit_array.iter() {
        two_to_the_bit_array.push(2_f64.powf(i));
    }
    (bit_array, two_to_the_bit_array)
}

fn init_weight_array(exponent: usize, signal_length: usize) -> Vec<f64> {
    let fexponent: f64 = exponent as f64;
    let fsignal_length: f64 = signal_length as f64;
    let mut weight_array = Vec::new();

    let mut fi = 0.0;
    while fi < fsignal_length {
        weight_array.push(2_f64.powf(
            ((fexponent * fi) / fsignal_length).ceil() - ((fexponent * fi) / fsignal_length),
        ));
        fi += 1.0;
    }
    weight_array
}

fn init_fft(signal_length: usize) -> (Arc<dyn RealToComplex<f64>>, Arc<dyn ComplexToReal<f64>>) {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal_length);
    let ifft = planner.plan_fft_inverse(signal_length);
    (fft, ifft)
}

// fn signalize(mut num_to_signalize: usize, two_to_the_bit_array: &[f64]) -> Vec<f64> {
//     let mut signal = vec![0.0_f64; two_to_the_bit_array.len()];
//     let mut i = 0;
//     while num_to_signalize > 0 {
//         signal[i] = num_to_signalize as f64 % two_to_the_bit_array[i];
//         num_to_signalize /= two_to_the_bit_array[i] as usize;
//         i += 1;
//     }
//     signal
// }

// fn designalize(signal: &[f64], two_to_the_bit_array: &[f64]) -> i64 {
//     let mut base = 1;
//     let mut resultant_num = 0;
//     for i in 0..signal.len() {
//         resultant_num += signal[i] as i64 * base;
//         base *= two_to_the_bit_array[i] as i64;
//     }
//     resultant_num
// }

/// Parse the given string as a `u128`.
fn parsenum(s: String) -> usize {
    s.parse().unwrap_or_else(|_| usage())
}

fn check_roundoff(roundoff: &f64, i: &usize) -> f64 {
    if *roundoff > 0.4375 {
        eprintln!(
            "Roundoff error is too great: {:.4} at iteration {:?}. Try a higher signal length.",
            roundoff, i
        );
        std::process::exit(1);
    }
    *roundoff
}

/// Print a usage error message and exit.
fn usage() -> ! {
    eprintln!(
        "Usage: toy-gimps-in-rust -e <exponent> [-s <signal_length>] [-f <update_frequency>] [-v]"
    );
    std::process::exit(1);
}

/// Print a help message and exit.
fn usage_help() -> ! {
    println!("Usage: toy-gimps-in-rust -e <exponent> [-s <signal_length>] [-f <update_frequency>] [-v] [-h]");
    println!("Checks if 2 ^ <exponent> - 1 is probably prime.");
    println!("  -e, --exponent <exponent>\t\tThe exponent to be checked");
    println!("  -s, --siglen <signal_length>\t\tThe signal length to use for the FFT");
    println!("  -f, --frequency <update_frequency>\tThe number of iterations between updates; defaults to <exponent> / 100");
    println!("  -v, --verbose\t\t\t\tRun with increased verbosity");
    println!("  -h, --help\t\t\t\tPrint this message and exit");
    std::process::exit(0);
}

// ----------------------------------------------------------------------------
// |                  Only tests from here down                               |
// ----------------------------------------------------------------------------

#[test]
fn test_gec() {
    fn prptest_with_forced_error(
        exponent: usize,
        signal_length: usize,
        forced_error: Vec<usize>,
    ) -> bool {
        let mut errors_list = forced_error.into_iter();
        let mut next_error;
        match errors_list.next() {
            Some(x) => next_error = x,
            None => panic!("No forced errors"),
        }

        let (_bit_array, two_to_the_bit_array) = init_bit_array(exponent, signal_length);
        let weight_array = init_weight_array(exponent, signal_length);
        let (fft, ifft) = init_fft(signal_length);
        let mut roundoff;
        let mut max_roundoff = 0.0;

        let mut residue = vec![0.0; signal_length];
        residue[0] = 3.0;
        residue = complete_carry(residue, &two_to_the_bit_array);

        let gec_l = (2000.min(((exponent as f64).sqrt() * 0.5).floor() as usize)).max(2);
        let gec_l_2 = gec_l.pow(2);
        let mut gec_d = residue.clone();
        let mut gec_prev_d = residue.clone();
        let mut gec_saved_d = residue.clone();
        let mut gec_saved_residue = residue.clone();
        let mut gec_saved_i: usize = 0;
        let mut three_signal = vec![0.0; signal_length];
        three_signal[0] = 3.0;
        println!("L: {:?}\nL^2: {:?}", gec_l, gec_l_2);

        let mut i = 0;
        while i < exponent {
            if i % gec_l == 0 && i != 0 {
                gec_prev_d = gec_d.clone();
                let (temp_gec_d, temp_roundoff) = multmod_with_ibdwt(
                    residue.clone(),
                    gec_d,
                    &two_to_the_bit_array,
                    &weight_array,
                    &fft,
                    &ifft,
                    false,
                );
                gec_d = temp_gec_d;
                check_roundoff(&temp_roundoff, &i);
            }
            if (i % gec_l_2 == 0 && i != 0) || (i % gec_l == 0 && i + gec_l >= exponent) {
                let mut check_value = gec_prev_d.clone();
                // println!("Iteration {:?}", i);
                // println!("Starting with prev_d = {:?}\nDesig: {:?}", gec_prev_d, designalize(&gec_prev_d, &two_to_the_bit_array));
                // println!("Starting with prev_d = {:?}", gec_prev_d);
                for _j in 0..gec_l {
                    let (temp_check_value, temp_roundoff) = squaremod_with_ibdwt(
                        check_value,
                        &two_to_the_bit_array,
                        &weight_array,
                        &fft,
                        &ifft,
                        false,
                    );
                    check_roundoff(&temp_roundoff, &i);
                    check_value = temp_check_value;
                }
                // println!("After {:?} squaremods, now have {:?}\nDesig: {:?}", gec_l, check_value, designalize(&check_value, &two_to_the_bit_array));
                // println!("After {:?} squaremods, now have {:?}", gec_l, check_value);
                let (temp_check_value, temp_roundoff) = multmod_with_ibdwt(
                    check_value,
                    three_signal.clone(),
                    &two_to_the_bit_array,
                    &weight_array,
                    &fft,
                    &ifft,
                    false,
                );
                check_roundoff(&temp_roundoff, &i);
                // println!("After multmod by three, now have {:?}\nDesig: {:?}", temp_check_value, designalize(&temp_check_value, &two_to_the_bit_array));
                // println!("Comparing it nwith d = {:?}\nDesig: {:?}\n", gec_d, designalize(&gec_d, &two_to_the_bit_array));
                // println!("After multmod by three, now have {:?}", temp_check_value);
                // println!("Comparing it nwith d = {:?}\n", gec_d);
                if temp_check_value != gec_d {
                    println!(
                        "Hardware error detected between iterations {:?} and {:?}; reverting to {:?}",
                        gec_saved_i, i, gec_saved_i
                    );
                    gec_d = gec_saved_d.clone();
                    residue = gec_saved_residue.clone();
                    i = gec_saved_i;
                } else {
                    gec_saved_d = gec_d.clone();
                    gec_saved_residue = residue.clone();
                    gec_saved_i = i;
                }
            }
            let (temp_residue, temp_roundoff) = squaremod_with_ibdwt(
                residue,
                &two_to_the_bit_array,
                &weight_array,
                &fft,
                &ifft,
                false,
            );
            residue = temp_residue;
            roundoff = check_roundoff(&temp_roundoff, &i);
            max_roundoff = roundoff.max(max_roundoff);

            if i == next_error {
                residue[0] += 1.0;
                match errors_list.next() {
                    Some(x) => next_error = x,
                    None => next_error = usize::MAX,
                }
            }

            i += 1;
        }

        let mut nine_signal = vec![0.0; signal_length];
        nine_signal[0] = 9.0;
        nine_signal = complete_carry(nine_signal, &two_to_the_bit_array);

        residue == nine_signal
    }
    let valid_primes = vec![
        (17, 4),
        (19, 4),
        (31, 4),
        (61, 8),
        (89, 8),
        (107, 8),
        (127, 8),
        (521, 32),
        (607, 32),
        (1279, 64),
        (23209, 2048),
    ];
    for (exponent, signal_length) in valid_primes {
        println!("Testing exponent {:?}", exponent);
        let forced_errors = vec![0, exponent / 3, exponent / 3 + 1, 2 * exponent / 3];
        println!("Should have errors at {:?}", forced_errors);
        assert!(
            prptest_with_forced_error(exponent, signal_length, forced_errors),
            "Failed at exponent {:?}",
            exponent
        );
        println!("Good\n");
    }
}

#[test]
fn test_real_fft() {
    let mut real_planner = RealFftPlanner::<f64>::new();
    for i in 0..23 {
        let signal_length = 1 << i;
        let fft = real_planner.plan_fft_forward(signal_length);
        let ifft = real_planner.plan_fft_inverse(signal_length);

        let mut example = Vec::new();
        for j in 0..signal_length {
            example.push(j as f64);
        }
        let start = example.clone();
        let mut fft_output = fft.make_output_vec();

        fft.process(&mut example, &mut fft_output).unwrap();
        ifft.process(&mut fft_output, &mut example).unwrap();
        example = example
            .iter()
            .map(|x| (x / example.len() as f64).round())
            .collect();
        println!("Comparing fft with ifft at sig len 2 ^ {:?}", i);
        if !(start == example) {
            println!("Failed comparison:\n{:?}\nand\n{:?}", start, example);
            panic!()
        } else {
            println!("Good");
        }
    }
}

#[test]
fn test_real_fft_problem_cases() {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let signal_length = 4;
    let fft = real_planner.plan_fft_forward(signal_length);
    let ifft = real_planner.plan_fft_inverse(signal_length);

    let mut example = vec![1.0, 5.0, 0.0, 0.0];
    let start = example.clone();
    let mut output = fft.make_output_vec();
    fft.process(&mut example, &mut output).unwrap();
    ifft.process(&mut output, &mut example).unwrap();
    example = example
        .iter()
        .map(|x| (*x / signal_length as f64).round())
        .collect();

    println!(
        "Non-squared case: Comparing\n{:?}\nwith\n{:?}",
        start, example
    );
    assert!(start == example);

    let squared: Vec<f64> = start.iter().map(|&x| x * x).collect();
    fft.process(&mut example, &mut output).unwrap();
    output = output
        .iter()
        .map(|&x| (x * x) / signal_length as f64)
        .collect();
    ifft.process(&mut output, &mut example).unwrap();
    example = example.iter().map(|x| (*x).round()).collect();

    println!(
        "Squared case: Comparing\n{:?}\nwith\n{:?}",
        squared, example
    );
    assert!(squared == example);
}
