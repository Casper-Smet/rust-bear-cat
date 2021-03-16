pub mod apriori;
pub mod classification;
pub mod euler;
pub mod math;
pub mod perceptron;
pub mod pi;
pub mod regression;
pub mod sorting;

#[test]
fn test_reg() {
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
fn test_knn() {
    let k = 2;
    let x: [[f32; 2]; 4] = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.06, 7.0]];
    let train: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [3.0, 3.0, 1.0],
        [4.0, 3.0, 1.0],
    ];
    let y_test: Vec<Option<f32>> = classification::k_nearest_neighbours(k, &x, &train);
    let y_true: Vec<Option<f32>> = vec![Some(0.0), Some(1.0), Some(1.0), Some(1.0)];
    assert_eq!(y_test, y_true);
}

#[test]
fn test_pi() {
    let pi: f64 = pi::pi_random_numbers(10000);
    assert!((std::f32::consts::PI - pi as f32).abs() < 1e-1);

    let pi: f64 = pi::pi_spigot_series(15);
    assert!((std::f32::consts::PI - pi as f32).abs() < 1e-1);

    let pi: f64 = pi::pi_gregory_leibniz(15);
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

#[test]
fn test_selection_sort() {
    let mut arr1: [i64; 4] = [21, -4, 6, 12];
    let arr1_sorted: [i64; 4] = [-4, 6, 12, 21];
    assert_eq!(sorting::selection_sort(&mut arr1), arr1_sorted);
}

#[test]
fn test_math() {
    assert!((math::Q_rsqrt(0.1) - 1. / (0.1 as f32).sqrt()).abs() < (0.1 as f32));
    assert!((math::Q_rsqrt(1.5) - 1. / (1.5 as f32).sqrt()).abs() < (0.1 as f32));
    assert!((math::Q_rsqrt(100.) - 1. / (100. as f32).sqrt()).abs() < (0.1 as f32));

    assert_eq!(math::factorial(0), 1);
    assert_eq!(math::factorial(1), 1);
    assert_eq!(math::factorial(3), 1 * 2 * 3);
    assert_eq!(math::factorial(4), 1 * 2 * 3 * 4);
}

#[test]
fn test_perceptron() {
    let p_and = perceptron::Node {
        weights: vec![0.5, 0.5],
        bias: -1.,
    };
    let input1 = vec![0.0, 0.0];
    let input2 = vec![0.0, 1.0];
    let input3 = vec![1.0, 0.0];
    let input4 = vec![1.0, 1.0];

    // Test AND Perceptron
    assert_eq!(p_and.activate(&input1), 0.);
    assert_eq!(p_and.activate(&input2), 0.);
    assert_eq!(p_and.activate(&input3), 0.);
    assert_eq!(p_and.activate(&input4), 1.);

    let p_or = perceptron::Node {
        weights: vec![1., 1.],
        bias: -1.,
    };

    let l0 = perceptron::Layer {
        nodes: vec![p_and, p_or],
    };

    // Test AND and OR gates in one layer
    assert_eq!(l0.activate(&input1), vec![0., 0.]);
    assert_eq!(l0.activate(&input2), vec![0., 1.]);
    assert_eq!(l0.activate(&input3), vec![0., 1.]);
    assert_eq!(l0.activate(&input4), vec![1., 1.]);

    let p_1 = perceptron::Node {
        weights: vec![1., -1.],
        bias: -1.,
    };
    let p_2 = perceptron::Node {
        weights: vec![-1., 1.],
        bias: -1.,
    };
    let p_3 = perceptron::Node {
        weights: vec![1., 1.],
        bias: -2.,
    };
    let p_4 = perceptron::Node {
        weights: vec![0., 0., 1.],
        bias: -1.,
    };
    let p_5 = perceptron::Node {
        weights: vec![1., 1., 0.],
        bias: -1.,
    };

    let l1 = perceptron::Layer {
        nodes: vec![p_1, p_2, p_3],
    };
    let l2 = perceptron::Layer {
        nodes: vec![p_4, p_5],
    };
    let n1 = perceptron::Network {
        layers: vec![l1, l2],
    };

    // Test half adder
    assert_eq!(n1.activate(input1), vec![0., 0.]);
    assert_eq!(n1.activate(input2), vec![0., 1.]);
    assert_eq!(n1.activate(input3), vec![0., 1.]);
    assert_eq!(n1.activate(input4), vec![1., 0.]);
}

#[test]
fn test_euler() {
    let e = euler::e_approximation(30);
    assert!((std::f32::consts::E - e as f32).abs() < 1e-1);
    let e = euler::random_e(1000);
    assert!((std::f32::consts::E - e as f32).abs() < 1e-1);
}
