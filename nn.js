class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes, sequential) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.model = sequential ? sequential : this.createModel();
  }

  createModel() {
    const model = tf.sequential();

    // Hidden Layer
    const hidden = tf.layers.dense({
      units: this.hidden_nodes,
      inputShape: [this.input_nodes],
      activation: "sigmoid"
    });
    model.add(hidden);

    // Output Layer
    const output = tf.layers.dense({
      units: this.output_nodes,
      activation: "softmax"
    });
    model.add(output);

    return model;
  }

  predict(inputs) {
    return tf.tidy(() => {
      const xs = tf.tensor2d([inputs]);
      const ys = this.model.predict(xs);
      xs.dispose();
      const outputs = ys.dataSync();
      ys.dispose();
      return outputs;
    });
  }

  copy() {
    return tf.tidy(() => {
      const modelCopy = this.createModel();
      const weights = this.model.getWeights();
      let weightCopies = [];
      for (let i in weights) {
        weightCopies[i] = weights[i].clone();
      }
      modelCopy.setWeights(weightCopies);
      return new NeuralNetwork(
        this.input_nodes,
        this.hidden_nodes,
        this.output_nodes,
        modelCopy
      );
    });
  }

  mutate(rate) {
    tf.tidy(() => {
      const weights = this.model.getWeights();
      const mutatedWeights = [];
      for (let i in weights) {
        let tensor = weights[i];
        let shape = weights[i].shape;
        let values = tensor.dataSync().slice();
        for (let j in values) {
          if (chance.floating({ min: 0, max: 1 }) < rate) {
            let w = values[j];
            values[j] = w + chance.floating({ min: -1, max: 1 });
          }
        }

        const newTensor = tf.tensor(values, shape);
        mutatedWeights[i] = newTensor;
      }
      this.model.setWeights(mutatedWeights);
    });
  }
}
