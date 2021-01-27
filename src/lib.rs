#[cfg(test)]
mod apriori;
#[cfg(test)]
mod classification;
#[cfg(test)]
mod pi;
#[cfg(test)]
mod regression;

#[test]
fn reg() {
    let x = 30.0;
    let a = 65.14157;
    let b = 0.385225;
    let y = a + x * b;

    println!("{} + {} * {} = {}", a, b, x, y);

    let age: Vec<f32> = vec![43.0, 21.0, 25.0, 42.0, 57.0, 59.0];
    let glucose: Vec<f32> = vec![99.0, 65.0, 79.0, 75.0, 87.0, 81.0];
    let func = regression::linear_regression(&age, &glucose);
    let y_test = func(x);
    println!("{}", y_test);
    assert_eq!(y, y_test);
    let age: [f32; 6] = [43.0, 21.0, 25.0, 42.0, 57.0, 59.0];
    let glucose: [f32; 6] = [99.0, 65.0, 79.0, 75.0, 87.0, 81.0];
    let func = regression::linear_regression(&age, &glucose);
    let y_test = func(x);
    println!("{}", y_test);
    assert_eq!(y, y_test);
}
#[test]
fn knn() {
    let k = 2;
    let x: [[f32; 2]; 4] = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.06, 7.0]];
    let train: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [3.0, 3.0, 1.0],
        [4.0, 3.0, 1.0],
    ];
    let y_test: Vec<f32> = classification::k_nearest_neighbours(k, &x, &train);
    let y_true: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0];
    assert_eq!(y_test, y_true);
}

#[test]
fn test_pi() {
    let pi: f64 = pi::pi_random_numbers(10000);
    assert!((std::f32::consts::PI - pi as f32).abs() < 1e-1);
}

#[test]
fn test_apriori() {
    use std::collections::HashSet;

    let transactions: Vec<HashSet<String>> = apriori::read_file("./data/store_data.csv");
    println!("{:?}", transactions[4]);
    println!("{:?}", transactions[10]);

    let sup: f32 = apriori::support(transactions[4].iter().cloned().collect(), &transactions);
    println!("Support:    {}", sup);
    let conf: f32 = apriori::confidence(
        transactions[4].iter().cloned().collect(),
        transactions[10].iter().cloned().collect(),
        &transactions,
    );
    println!("Confidence: {}", conf);
    let lift: f32 = apriori::lift(
        transactions[4].iter().cloned().collect(),
        transactions[10].iter().cloned().collect(),
        &transactions,
    );
    println!("Lift:       {}", lift);

    let (max_lift, max_prod) =
        apriori::best_lift(transactions[4].iter().cloned().collect(), &transactions);
    println!("Max Lift:   {} with {:?}", max_lift, max_prod);
}
