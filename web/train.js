const fs = require('fs');

const facemesh = require('@tensorflow-models/facemesh');
const posenet = require('@tensorflow-models/posenet');
const blazeface = require('@tensorflow-models/blazeface');
const Stats = require('stats.js')
// const tfjs = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
// const tfjsWasm = require('@tensorflow/tfjs-backend-wasm')
const {loadImage } = require('canvas');
const util = require('./util');

const state = {
    backend: 'cpu', //'cpu'
    maxFaces: 1
};

let truncatedMobileNet;

// This is our program. This time we use JavaScript async / await and promises to handle asynchronicity.
(async () => {
  const buf = fs.readFileSync('./lena.jpg');
  image = tf.node.decodeJpeg(buf, 3);
  const facialFeature = extractFacialFeature(image);
//   faceMeshPrediction[0].mesh.array().then(array => console.log(array))
//   faceMeshPrediction[0].scaledMesh.array().then(array => console.log(array))
  
//   console.log(faceMeshPrediction[0].scaledMesh.array())
  
  // model, use tfjs functional api.
//   const inputFace = tf.input({shape: [784]});
//   const flattenFace = tf.layers.flatten({inputShape: truncatedMobileNet.outputs[0].shape.slice(1)})
//   const denseFace1 = tf.layers.dense({units: 64, activation: 'relu', kernelInitializer: 'varianceScaling',}).apply(flattenFace);

//   const inputMobileNet = tf.input({shape: [784]});
  
  
//   const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
//   const model = tf.model({inputs: input, outputs: dense2});
})();

// Extract Pose Feature.
async function extractPoseFeature(image, scoreThreshold){
  const maxDistance = Math.pow((image.shape[0]), 2) + Math.pow((image.shape[1]), 2);
  poseNet = await posenet.load({
    architecture: 'MobileNetV1',
    quantBytes: 1
  });
  const pose = await poseNet.estimateSinglePose(image, {
    flipHorizontal: false,
    scoreThreshold: 0.7
  });
  const upperPoseCount = 11
  pose = pose.keypoints.slice(0, upperPoseCount) // only getting coordindate of upper body.
  let keypointMetric = new Array(upperPoseCount);
  for (let i = 0; i < upperPoseCount; i++) {
    keypointMetric[i] = new Array(upperPoseCount);
    for (let j = 0; j < upperPoseCount; j++) {
        if (pose[i].score <= scoreThreshold){
            keypointMetric[i][j] = 1;                
        }
        else{
            keypointMetric[i][j] = Math.pow((pose[i].position.x - pose[j].position.x), 2) + Math.pow((pose[i].position.y - pose[j].position.y), 2) / maxDistance;
        }
    }
  }
  return tf.tensor(keypointMetric);
}

// Extract facial feature. First run bazelface to crop face, 
// then pass cropped face area to MobileNet to extract feature.
// bazelface: https://github.com/tensorflow/tfjs-models/tree/master/blazeface
async function extractFacialFeature(image){
  // Extract image embedding from MobileNet.
  const blazefaceModel = await blazeface.load();
  const blazefacePredictions = await blazefaceModel.estimateFaces(image, {returnTensors: true});

  // Crop and Resize to Mobile net input size.
  const boxes = blazefacePredictions[0].topLeft.concat(blazefacePredictions[0].bottomRight)
  const croppedFace = tf.image.cropAndResize(image.expandDims(0), boxes.expandDims(0), [0], [224, 224])
  truncatedMobileNet = await loadTruncatedMobileNet();
  const embeddings = truncatedMobileNet.predict(croppedFace);
  embeddings.print();
  return embeddings;
}

// mobilenet: https://github.com/tensorflow/tfjs-models/tree/master/mobilenet
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}