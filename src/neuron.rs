#[derive(Debug)]
pub struct Node {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub learning_rate: f32,
    output: f32,
    error: f32,
    delta_weights: Vec<f32>,
    delta_bias: f32,
    gradients: Vec<f32>,
}

impl Node {
    pub fn new(weights: Vec<f32>, bias: f32, learning_rate: f32) -> Node {
        Node {
            weights: weights,
            bias: bias,
            learning_rate: learning_rate,
            output: 0.,
            error: 0.,
            delta_weights: Vec::new(),
            delta_bias: 0.,
            gradients: Vec::new(),
        }
    }

    pub fn activate(&mut self, input: &Vec<f32>) -> f32 {
        if input.len() != self.weights.len() {
            panic!(
                "Input and weights have different lengths, got {} and {}",
                input.len(),
                self.weights.len()
            );
        }
        self.output = self.weights.iter().zip(input.iter()).map(|(a, b)| a * b).sum::<f32>() + self.bias;
        self.output = self.sigmoid(&self.output);
        self.output
    }

    pub fn sigmoid(&self, weighted_sum: &f32) -> f32 {
        1. / (1. + std::f32::consts::E.powf(-weighted_sum))
    }

    pub fn d_sigmoid(&self, output: &f32) -> f32 {
        output * (1. - output)
    }

    pub fn error_output(&mut self, target: &f32) {
        self.error = self.d_sigmoid(&self.output) * -(target - &self.output);
    }

    pub fn error_hidden(&mut self, next_weights: &Vec<f32>, next_errors: &Vec<f32>) {
        if next_weights.len() != next_errors.len() {
            panic!(
                "Weights and errors have different lengths, got {} and {}",
                next_weights.len(),
                next_errors.len()
            );
        }
        self.error = &self.output * (1. - &self.output) * next_weights.iter().zip(next_errors).map(|(w, e)| w * e).sum::<f32>();
    }

    pub fn calculate_gradients(&mut self, prev_outputs: &Vec<f32>) {
        if prev_outputs.len() != self.weights.len() {
            panic!(
                "Previous outputs and current weights have different lengths, got {} and {}",
                prev_outputs.len(),
                self.weights.len()
            );
        }
        self.gradients = prev_outputs.iter().map(|o| self.error * o).collect()
    }

    pub fn calculate_delta_weights(&mut self) {
        if self.weights.len() != self.gradients.len() {
            panic!(
                "Weights and gradients have different lengths, got {} and {}",
                self.weights.len(),
                self.gradients.len()
            );
        }
        self.delta_weights = self.gradients.iter().map(|g| &self.learning_rate * g).collect();
    }

    pub fn calculate_delta_bias(&mut self) {
        self.delta_bias = &self.learning_rate * &self.error;
    }

    pub fn update_weights(&mut self) {
        if self.weights.len() != self.delta_weights.len() {
            panic!(
                "Weights and delta weights have different lengths, got {} and {}",
                self.weights.len(),
                self.delta_weights.len()
            );
        }
        self.weights = self.weights.iter().zip(&self.delta_weights).map(|(w, d)| w - d).collect();
    }

    pub fn update_bias(&mut self) {
        self.bias -= self.delta_bias;
    }
}

pub struct Layer {
    pub nodes: Vec<Node>,
    activations: Vec<f32>,
    errors: Vec<f32>,
}

impl Layer {
    pub fn new(nodes: Vec<Node>) -> Layer {
        Layer {
            nodes: nodes,
            activations: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn activate(&mut self, input: &Vec<f32>) -> &Vec<f32> {
        for n in self.nodes.iter() {
            if n.weights.len() != input.len() {
                panic!(
                    "Node weights and inputs have different lengths, got {} and {}",
                    n.weights.len(),
                    input.len()
                );
            }
        }
        self.activations = self.nodes.iter_mut().map(|n| n.activate(input)).collect::<Vec<f32>>();
        &self.activations
    }

    pub fn weights(&self) -> Vec<Vec<f32>> {
        self.nodes.iter().map(|n| n.weights.clone()).collect()
    }

    pub fn errors_output(&mut self, target: &Vec<f32>) {
        if self.nodes.len() != target.len() {
            panic!(
                "Input and target have different lengths, got {} and {}",
                self.nodes.len(),
                target.len()
            );
        }
        self.nodes.iter_mut().zip(target).for_each(|(n, t)| n.error_output(t));
        self.errors = self.nodes.iter().map(|n| n.error).collect();
    }

    pub fn errors_hidden(&mut self, weights: &Vec<Vec<f32>>, errors: &Vec<f32>) {
        for w in weights {
            if self.nodes.len() != w.len() {
                panic!(
                    "Number of nodes is not equal to number of weights of node in next layer, got {} and {}",
                    self.nodes.len(),
                    w.len()
                );
            }
        }
        self.nodes
            .iter_mut()
            .enumerate()
            .for_each(|(i, n)| n.error_hidden(&weights.iter().map(|k| k[i]).collect::<Vec<f32>>(), &errors));
        self.errors = self.nodes.iter().map(|n| n.error).collect();
    }

    pub fn calculcate_gradients(&mut self, prev_outputs: &Vec<f32>) {
        for n in self.nodes.iter() {
            if n.weights.len() != prev_outputs.len() {
                panic!(
                    "Node weights and previous outputs have different lengths, got {} and {}",
                    n.weights.len(),
                    prev_outputs.len()
                );
            }
        }
        self.nodes.iter_mut().for_each(|n| n.calculate_gradients(prev_outputs));
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
    output: Vec<f32>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Network {
        Network {
            layers: layers,
            output: Vec::new(),
        }
    }

    pub fn activate(&mut self, input: &Vec<f32>) -> &Vec<f32> {
        let mut output = input;
        for l in self.layers.iter_mut() {
            output = l.activate(&output);
        }
        self.output = output.to_vec();
        &self.output
    }

    pub fn backprop(&mut self, input: &Vec<f32>, target: &Vec<f32>) {
        for i in (0..self.layers.len()).rev() {
            let prev_output = if i == 0 {
                input.clone()
            } else {
                self.layers[i - 1].activations.clone()
            };
            if i == self.layers.len() - 1 {
                self.layers[i].errors_output(&target);
            } else {
                let next_weights = self.layers[i + 1].weights().clone();
                let next_errors = self.layers[i + 1].errors.clone();

                self.layers[i].errors_hidden(&next_weights, &next_errors);
            }
            self.layers[i].calculcate_gradients(&prev_output);
        }

        for layer in 0..self.layers.len() {
            for node in 0..self.layers[layer].nodes.len() {
                self.layers[layer].nodes[node].calculate_delta_weights();
                self.layers[layer].nodes[node].calculate_delta_bias();
                self.layers[layer].nodes[node].update_weights();
                self.layers[layer].nodes[node].update_bias();
            }
        }
    }

    pub fn epoch(&mut self, inputs: &Vec<&Vec<f32>>, targets: &Vec<&Vec<f32>>) {
        if inputs.len() != targets.len() {
            panic!(
                "Inputs and targets have different lengths, got {} and {}",
                inputs.len(),
                targets.len()
            );
        }
        for (target, input) in targets.iter().zip(inputs) {
            self.activate(input);
            self.backprop(input, target);
        }
    }

    pub fn train(&mut self, inputs: &Vec<&Vec<f32>>, targets: &Vec<&Vec<f32>>, epochs: u128) {
        for _ in 0..epochs {
            self.epoch(inputs, targets);
        }
    }
}
