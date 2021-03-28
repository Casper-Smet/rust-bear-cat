pub fn primes_count_sieve(n: usize) -> u64 {
    // counts the amount of primes between 0 and n using while loops
    let mut sieve: Vec<bool> = (0..n+1).map(|_| true).collect();
    let mut p: usize = 2;

    while p * p <= n {

        // check if current number is prime
        if sieve[p] {
            // Update all multiples of p
            let mut i = p * 2;
            while i < n + 1 {
                sieve[i] = false;
                i += p;
            }
        }
        p += 1;
    }
    // set 0 and 1 to false
    sieve[0] = false;
    sieve[1] = false;
    // sum things
    sieve.iter().fold(0, |acc, is_prime| {
        if *is_prime {acc + 1} else {acc}
    })
}
