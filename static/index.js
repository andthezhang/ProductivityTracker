const model_path = "https://storage.googleapis.com/productivemodel/static/model.json"
// model.js ======================================================================
async function extractPoseFeature(image, poseNet, scoreThreshold=0.6, verbose=false){
  const upperPoseCount = 11;
  const maxDistance = Math.pow((image.shape[0]), 2) + Math.pow((image.shape[1]), 2);
  const pose = await poseNet.estimateSinglePose(image, {
    flipHorizontal: true
  });
  let upperPose = pose.keypoints.slice(0, upperPoseCount) // only getting coordindate of upper body.
  let distanceKernel = new Array(upperPoseCount);
  let cosineKernel = new Array(upperPoseCount);

  for (let i = 0; i < upperPoseCount; i++) {
    distanceKernel[i] = new Array(upperPoseCount);
    cosineKernel[i] = new Array(upperPoseCount);
    for (let j = 0; j < upperPoseCount; j++) {
        if (i == j){
          distanceKernel[i][j] = 0;
        }
        else if (upperPose[i].score >= scoreThreshold && upperPose[j].score >= scoreThreshold){
          distanceKernel[i][j] = (Math.pow((upperPose[i].position.x - upperPose[j].position.x), 2) + Math.pow((upperPose[i].position.y - upperPose[j].position.y), 2) )/ maxDistance;
        }
        else{
          distanceKernel[i][j] = 1;
        }
        if (i == j){
          cosineKernel[i][j] = 1;
        }
        else{
          cosineKernel[i][j] = (upperPose[i].position.x - upperPose[j].position.x) / Math.sqrt((Math.pow((upperPose[i].position.x - upperPose[j].position.x), 2) + Math.pow((upperPose[i].position.y - upperPose[j].position.y), 2) ))
        }
    }
  }
  const distanceKernelTensor = tf.tensor(distanceKernel);
  const cosineKernelTensor = tf.tensor(cosineKernel);

  // Concat two kernels and return.
  const returnTensor = tf.concat([distanceKernelTensor, cosineKernelTensor]).expandDims(0);

  if (verbose){
    console.log("extractPoseFeature=======")
    console.log(returnTensor.shape)
    // distanceKernelTensor.print();
    // cosineKernelTensor.print();
    // returnTensor.print()
  }

  return returnTensor;
}
// mobilenet: https://github.com/tensorflow/tfjs-models/tree/master/mobilenet
// This function is adapted from https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// Extract facial feature. First run bazelface to crop face, 
// then pass cropped face area to MobileNet to extract feature.
// bazelface: https://github.com/tensorflow/tfjs-models/tree/master/blazeface
async function extractFaceFeature(image, blazefaceModel, truncatedMobileNet, verbose=false){
  // Extract image embedding from MobileNet.
  // const blazefaceModel = await blazeface.load();
  const blazefacePredictions = await blazefaceModel.estimateFaces(image, {returnTensors: true});
  if (blazefacePredictions == null || blazefacePredictions.length == 0 ){
    return null;
  }
  // Crop and Resize to Mobile net input size.
  const boxes = blazefacePredictions[0].topLeft.concat(blazefacePredictions[0].bottomRight);
  const croppedFace = tf.image.cropAndResize(image.expandDims(0), boxes.expandDims(0), [0], [224, 224]);
  const embeddings = truncatedMobileNet.predict(croppedFace);
  if (verbose){
    console.log("extractFaceFeature=======")
    console.log(embeddings.shape)
  }
  return embeddings;
}

async function loadCustomModel(){
const model = await tf.loadGraphModel('https://storage.googleapis.com/productivemodel/static/model.json')
// const model = await tf.loadGraphModel('http://localhost:8080/no_pose_affectnet_model_tfjs/model.json');
return model;
}
async function extractFeature(image, truncatedMobileNet, verbose=true){
const resizeImage = tf.image.resizeBilinear(image.expandDims(0), [224, 224]);
const xs = await truncatedMobileNet.predict(resizeImage);
return xs;
}

// ui.js =================================================
function displayPrediction(boredPredictions) {     
  boredPredictions.data().then(data => {
      const prob = data[0];
      console.log(prob);
      console.log(typeof prob);
      
      document.getElementById('boredProb').innerHTML = prob.toFixed(4);
      }
  )
}
var chart = new CanvasJS.Chart("chartContainer", {
title: {
  text: "Your Productivity Level"
},
axisY: {
      title: "Probability of being productive",
      maximum: 100,
  includeZero: true,
  suffix: " %"
},
data: [{
  type: "column",	
  indexLabel: "{y}",
  dataPoints: [
          { label: "Productive Level", y: 50 },
  ]
}]
});
async function updateChart(boredPredictions) {
  boredPredictions.data().then(data => {
      const prob = data[1].toFixed(4)*100;
      const newColor = prob <= 30 ? "#FF2500" : prob <= 50 ? "#FF6000" : prob > 50 ? "#6B8E23 " : null;
      chart.options.data[0].dataPoints = [{label: "Productive Level", y: prob, color: newColor}]
      chart.render();
      }
  )
};

// index.js ==============================================

const stats = new Stats();
stats.showPanel(0);
document.body.prepend(stats.domElement);

let model, blazefaceModel, truncatedMobileNet;
let ctx, videoWidth, videoHeight, video, canvas;

const state = {
  backend: 'wasm'
};

const gui = new dat.GUI();
gui.add(state, 'backend', ['wasm', 'webgl', 'cpu']).onChange(async backend => {
  await tf.setBackend(backend);
});

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  stats.begin();

  const returnTensors = false;
  const flipHorizontal = true;
  const annotateBoxes = true;
  const predictions = await blazefaceModel.estimateFaces(
    video, returnTensors, flipHorizontal, annotateBoxes);
  const faceFeature = await extractFaceFeature(tf.browser.fromPixels(video), blazefaceModel, truncatedMobileNet);
  if (faceFeature != null){
    const feature = await extractFeature(tf.browser.fromPixels(video), truncatedMobileNet);
    const boredPredictions = await model.predict([faceFeature, feature]);
    await updateChart(boredPredictions);
  }

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
        if (annotateBoxes) {
          predictions[i].landmarks = predictions[i].landmarks.arraySync();
        }
      }

      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];
      ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
      ctx.fillRect(start[0], start[1], size[0], size[1]);

      if (annotateBoxes) {
        const landmarks = predictions[i].landmarks;

        ctx.fillStyle = "blue";
        for (let j = 0; j < landmarks.length; j++) {
          const x = landmarks[j][0];
          const y = landmarks[j][1];
          ctx.fillRect(x, y, 5, 5);
        }
      }
    }
  }

  stats.end();

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;
  
  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  // Load models.
  blazefaceModel = await blazeface.load();
  // poseNet = await posenet.load({
  //   architecture: 'MobileNetV1',
  //   quantBytes: 1
  // });
  truncatedMobileNet = await loadTruncatedMobileNet();
  model = await loadCustomModel();

  renderPrediction();
};

setupPage();
