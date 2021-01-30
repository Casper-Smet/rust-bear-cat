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
