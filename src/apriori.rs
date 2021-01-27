use std::collections::HashSet;
use std::fs;

pub fn read_file(filepath: &str) -> Vec<HashSet<String>> {
    let output = fs::read_to_string(filepath).expect("Something went wrong reading the file");

    let transactions: Vec<&str> = output.split("\n").collect();
    let mut cleaned_transactions = Vec::new();
    // Skip first line containing all products
    for t in transactions.iter().skip(1) {
        if t == &"" {
            break; // Skip empty transactions
        }
        let set = t.split(",").collect::<HashSet<&str>>();
        let set = set.iter().map(|x| x.to_string()).collect();
        cleaned_transactions.push(set);
    }

    cleaned_transactions
}

pub fn support(products: HashSet<String>, transactions: &Vec<HashSet<String>>) -> f32 {
    let count = transactions
        .iter()
        .filter(|&t| products.intersection(&t).collect::<HashSet<_>>().len() == products.len()) // All p in products are in t
        .count();

    count as f32 / transactions.len() as f32
}

pub fn confidence(
    products_a: HashSet<String>,
    products_b: HashSet<String>,
    transactions: &Vec<HashSet<String>>,
) -> f32 {
    let products_c = products_a.iter().cloned().collect();
    let prod_union = products_a.into_iter().chain(products_b).collect();
    support(prod_union, &transactions) / support(products_c, &transactions)
}

pub fn lift(
    products_a: HashSet<String>,
    products_b: HashSet<String>,
    transactions: &Vec<HashSet<String>>,
) -> f32 {
    let products_a_ = products_a.iter().cloned().collect();
    let products_b_ = products_a.iter().cloned().collect();
    let prod_union = products_a.into_iter().chain(products_b).collect();
    support(prod_union, &transactions)
        / (support(products_a_, &transactions) * support(products_b_, &transactions))
}

pub fn best_lift(
    given_product: HashSet<String>,
    transactions: &Vec<HashSet<String>>,
) -> (f32, HashSet<String>) {
    let mut unique_products = HashSet::new();
    let mut max_lift: f32 = 0.0;
    let mut max_prod = HashSet::new();

    for t in transactions {
        // Rewrite to reduce with union operator
        unique_products.extend(t);
    }

    let given_product_str: Vec<_> = given_product.iter().cloned().collect();
    unique_products.remove(&given_product_str[0]); // Only works for given_product with 1 element

    for tr_product in unique_products.iter() {
        let mut set_product = HashSet::new();
        set_product.insert(tr_product.to_string());

        let current_lift = lift(
            given_product.iter().cloned().collect(),
            set_product.iter().cloned().collect(),
            transactions,
        );

        if current_lift >= max_lift {
            max_lift = current_lift;
            max_prod = set_product.iter().cloned().collect();
        }
    }
    return (max_lift, max_prod);
}
