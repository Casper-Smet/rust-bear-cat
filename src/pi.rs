use rand::Rng;

pub fn pi_random_numbers(iters: i64) -> f64 {
    let mut rng = rand::thread_rng();
    let mut total: i64 = 0;

    for _ in 0..iters {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);

        if x.hypot(y) <= 1.0 {
            total += 1;
        }
    }
    (total as f64 / iters as f64) * 4.0 // Return Pi
}
