#[derive(Debug)]
pub struct Node {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub learning_rate: f32,
}

impl Node {
    pub fn activate(&self, input: &Vec<f32>) -> f32 {
        if input.len() != self.weights.len() {
            panic!(
                "Input and weights have different lengths, got {} and {}",
                input.len(),
                self.weights.len()
            );
        }
        let total = self.weights.iter().zip(input.iter()).map(|(a, b)| a * b).sum::<f32>() + self.bias;
        self.sigmoid(&total)
    }

    pub fn sigmoid(&self, weighted_sum: &f32) -> f32 {
        1. / (1. + std::f32::consts::E.powf(-weighted_sum))
    }
    pub fn d_sigmoid(&self, output: &f32) -> f32 {
        output * (1. - output)
    }

    pub fn error_output(&self, output: &f32, target: &f32) -> f32 {
        self.d_sigmoid(output) * -(target - output)
    }
    pub fn error_hidden(&self, output: &f32, next_weights: &Vec<f32>, next_errors: &Vec<f32>) -> f32 {
        if next_weights.len() != next_errors.len() {
            panic!(
                "Weights and errors have different lengths, got {} and {}",
                next_weights.len(),
                next_errors.len()
            );
        }
        output * next_weights.iter().zip(next_errors).map(|(w, e)| w * e).sum::<f32>()
    }

    pub fn calculate_delta_weights(&self, inputs: &Vec<f32>, error: &f32) -> Vec<f32> {
        inputs.iter().map(|w| self.learning_rate * error * w).collect()
    }

    pub fn calculate_delta_bias(&self, error: &f32) -> f32 {
        self.learning_rate * error
    }

    pub fn update_weights(&mut self, delta_weights: &Vec<f32>) {
        if self.weights.len() != delta_weights.len() {
            panic!(
                "Weights and delta weights have different lengths, got {} and {}",
                self.weights.len(),
                delta_weights.len()
            );
        }
        self.weights = self.weights.iter().zip(delta_weights).map(|(w, d)| w - d).collect();
    }

    pub fn update_bias(&mut self, delta_bias: &f32) {
        self.bias = self.bias - delta_bias;
    }

    // MSE Loss
    pub fn loss(&mut self, inputs: &Vec<&Vec<f32>>, targets: &Vec<f32>) -> f32 {
        if inputs.len() != targets.len() {
            panic!(
                "Inputs and targets have different lengths, got {} and {}",
                inputs.len(),
                targets.len()
            );
        }
        let outputs: Vec<f32> = inputs.iter().map(|i| self.activate(i)).collect();
        outputs
            .iter()
            .zip(targets)
            .map(|(i, t)| self.error_output(&i, t).powf(2.))
            .sum::<f32>()
            / targets.len() as f32
    }

    pub fn epoch(&mut self, inputs: &Vec<&Vec<f32>>, targets: &Vec<f32>) -> f32 {
        if inputs.len() != targets.len() {
            panic!(
                "Inputs and targets have different lengths, got {} and {}",
                inputs.len(),
                targets.len()
            );
        }
        for (input, target) in inputs.iter().zip(targets) {
            let output = self.activate(input);
            let error = self.error_output(&output, target);
            let delta_weights = self.calculate_delta_weights(input, &error);
            let delta_bias = self.calculate_delta_bias(&error);
            self.update_weights(&delta_weights);
            self.update_bias(&delta_bias);
        }
        self.loss(inputs, targets)
    }
}

pub struct Layer {
    pub nodes: Vec<Node>,
}

impl Layer {
    pub fn activate(&self, input: &Vec<f32>) -> Vec<f32> {
        self.nodes.iter().map(|n| n.activate(input)).collect()
    }
    pub fn weights(&self) -> Vec<&Vec<f32>> {
        self.nodes.iter().map(|n| &n.weights).collect()
    }

    pub fn errors_output(&self, output: &Vec<f32>, target: &Vec<f32>) -> Vec<f32> {
        if output.len() != target.len() {
            panic!("Input and target have different lengths, got {} and {}", output.len(), target.len());
        }
        self.nodes
            .iter()
            .zip(output)
            .zip(target)
            .map(|((n, o), t)| n.error_output(&o, t))
            .collect()
    }
    pub fn errors_hidden(&self, output: &Vec<f32>, weights: &Vec<&Vec<f32>>, errors: &Vec<f32>) -> Vec<f32> {
        if self.nodes.len() != output.len() {
            panic!(
                "Number of nodes is not equal to length of output, got {} and {}",
                self.nodes.len(),
                output.len()
            );
        }
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
            .iter()
            .zip(output)
            .enumerate()
            .map(|(i, (n, o))| n.error_hidden(&o, &weights.iter().map(|k| k[i]).collect::<Vec<f32>>(), &errors))
            .collect()
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
    pub fn activate_full(&self, input: &Vec<f32>) -> Vec<Vec<f32>> {
        let mut outputs = vec![input.to_vec()];
        for (i, l) in self.layers.iter().enumerate() {
            outputs.push(l.activate(&outputs[i]));
        }
        outputs
    }

    pub fn backprop(&mut self, input: &Vec<f32>, target: &Vec<f32>) {
        let output = self.activate(input);
        let activations = self.activate_full(&input);
        &self.layers.reverse();
        let weights = &self.layers.iter().map(|l| l.weights()).collect::<Vec<Vec<&Vec<f32>>>>();
        let mut errors = vec![self.layers[0].errors_output(&output, &target)];

        for i in 1..self.layers.len() {
            errors.push(self.layers[i].errors_hidden(&activations[i], &weights[i - 1], &errors[i - 1]));
        }

        for layer in 0..self.layers.len() {
            for node in 0..self.layers[layer].nodes.len() {
                let delta_weights = self.layers[layer].nodes[node].calculate_delta_weights(input, &errors[layer][node]);
                let delta_bias = self.layers[layer].nodes[node].calculate_delta_bias(&errors[layer][node]);
                self.layers[layer].nodes[node].update_weights(&delta_weights);
                self.layers[layer].nodes[node].update_bias(&delta_bias);
            }
        }
        &self.layers.reverse(); // Restore inplace reverse
    }

    pub fn epoch(&mut self, inputs: &Vec<&Vec<f32>>, targets: &Vec<&Vec<f32>>) {
        if inputs.len() != targets.len() {
            panic!(
                "Inputs and targets have different lengths, got {} and {}",
                inputs.len(),
                targets.len()
            );
        }
        for (input, target) in inputs.iter().zip(targets) {
            self.backprop(input, target);
        }
    }
    pub fn train(&mut self, inputs: &Vec<&Vec<f32>>, targets: &Vec<&Vec<f32>>, epochs: u128) {
        for _ in 0..epochs {
            self.epoch(inputs, targets);
        }
    }
}
