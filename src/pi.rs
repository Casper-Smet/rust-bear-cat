use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub fn pi_random_numbers(iters: i64) -> f64 {
    let xs = Array::random(iters as usize, Uniform::new(-1., 1.));
    let ys = Array::random(iters as usize, Uniform::new(-1., 1.));

    let n_inside: i64 = xs
        .iter()
        .zip(ys.iter())
        .fold(0, |acc, (a, b)| acc + ((*a as f64).hypot(*b) <= 1.) as i64);

    (n_inside as f64 / iters as f64) * 4. // Return Pi
}

pub fn factorial_f64(n: f64) -> f64 {
    match n as u64 {
        0 | 1 => 1.,
        _ => factorial_f64(n - 1.) * n,
    }
}

fn single_iter_spigot(i: f64) -> f64 {
    let i_fact: f64 = factorial_f64(i);
    (i_fact * i_fact * (2. as f64).powf(i + 1.)) as f64 / factorial_f64(2. * i + 1.)
}

pub fn pi_spigot_series(i: u128) -> f64 {
    Array::range(0., i as f64, 1.0)
        .mapv(single_iter_spigot)
        .sum()
}

pub fn pi_gregory_leibniz(i: u128) -> f64 {
    4. * Array::range(0., i as f64, 1.)
        .mapv(|n: f64| (-1. as f64).powf(n) / (2. * n + 1.))
        .sum()
}
