#[derive(Debug)]
pub struct Node {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub learning_rate: f32,
}

impl Node {
    pub fn activate(&self, input: &Vec<f32>) -> f32 {
        let total = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + self.bias;
        self.sigmoid(&total)
    }

    pub fn sigmoid(&self, weighted_sum: &f32) -> f32 {
        1. / (1. + std::f32::consts::E.powf(-weighted_sum))
    }

    pub fn error(&self, input: &Vec<f32>, target: &f32) -> f32 {
        let output = self.activate(input);
        target - output
    }

    pub fn calculate_delta_weights(&self, input: &Vec<f32>, target: &f32) -> Vec<f32> {
        let error = self.error(input, target);
        let delta_weights = input
            .iter()
            .map(|i| self.learning_rate * error * i)
            .collect();
        delta_weights
    }

    pub fn calculate_delta_bias(&self, input: &Vec<f32>, target: &f32) -> f32 {
        let error = self.error(input, target);
        let delta_bias = self.learning_rate * error;
        delta_bias
    }

    pub fn update_weights(&mut self, delta_weights: &Vec<f32>) {
        self.weights = self
            .weights
            .iter()
            .zip(delta_weights)
            .map(|(w, d)| w + d)
            .collect();
    }

    pub fn update_bias(&mut self, delta_bias: &f32) {
        self.bias = self.bias + delta_bias;
    }

    // MSE Loss
    pub fn loss(&self, inputs: &Vec<&Vec<f32>>, targets: &Vec<f32>) -> f32 {
        inputs
            .iter()
            .zip(targets)
            .map(|(i, t)| self.error(i, t).powf(2.))
            .sum::<f32>()
            / targets.len() as f32
    }

    pub fn epoch(&mut self, inputs: &Vec<&Vec<f32>>, targets: &Vec<f32>) -> f32 {
        for (input, target) in inputs.iter().zip(targets) {
            let delta_weights = self.calculate_delta_weights(input, target);
            let delta_bias = self.calculate_delta_bias(input, target);
            self.update_weights(&delta_weights);
            self.update_bias(&delta_bias);
        }
        self.loss(inputs, targets)
    }
}

#[derive(Debug)]
pub struct Layer {
    pub nodes: Vec<Node>,
}

impl Layer {
    pub fn activate(&self, input: &Vec<f32>) -> Vec<f32> {
        self.nodes.iter().map(|n| n.activate(input)).collect()
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn activate(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.to_vec();
        for l in self.layers.iter() {
            output = l.activate(&output);
        }
        output
    }
}
