const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const util = require('./util');
const {ControllerDataset} = require('./dataset');
const {faceNetConfig} = require('./model/facenet')
const {poseNetConfig} = require('./model/posenet')



// Config.
const state = {
    backend: 'webgl', //'cpu'
    maxFaces: 1
};
tf.registerBackend(state.backend);
tf.ready()
const labels = ["true", "false"];
// tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tfjs-backend-wasm.wasm');
(async () => {
  // tf.setBackend(state.backend);
  const trainDataset = new ControllerDataset(labels);
  await trainDataset.loadFromNpy();
  train(trainDataset);
})();


// Train model with tfjs.
async function loadModel(){
  // MLP on Face Feature.
  const faceInput = tf.input({batchShape: faceNetConfig.outputBatchShape});
  const flattenFace = tf.layers.flatten().apply(faceInput);
  const denseFace = tf.layers.dense({units: 64, activation: 'relu', kernelInitializer: 'varianceScaling', useBias: true}).apply(flattenFace);

  // Pose Feature.
  const poseInput = tf.input({batchShape: poseNetConfig.outputBatchShape});
  const flattenPose = tf.layers.flatten().apply(poseInput);
  const densePose = tf.layers.dense({units: 64, activation: 'relu', kernelInitializer: 'varianceScaling', useBias: true}).apply(flattenPose);

  // Concat feature and pass to output layer.
  const concatFeature = tf.layers.concatenate().apply([denseFace, densePose]);
  const output = tf.layers.dense({units: 2, activation: 'softmax', kernelInitializer: 'varianceScaling', useBias: false}).apply(concatFeature);
  const model = tf.model({inputs: [faceInput, poseInput], outputs: output});

  return model;
}

async function train(controllerDataset){
  if (controllerDataset.xsFace == null) {
    throw new Error('Add some examples before training!');
  }
  const batchSize = 1
  const epochs = 1
  const model = await loadModel();
  const optimizer = tf.train.adam(0.001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy']});
  model.fit(controllerDataset.xsFace, controllerDataset.xsPose, controllerDataset.ys, {
    batchSize,
    epochs: epochs,
    shuffle: true,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log(batch)
        console.log('Loss: ' + logs.loss.toFixed(5))
      }
    }
  });
}