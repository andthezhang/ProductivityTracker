import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import {displayPrediction, updateChart} from './ui'
// import * as posenet from  '@tensorflow-models/posenet';

import {extractFaceFeature, extractFeature, loadTruncatedMobileNet, loadCustomModel} from './model';

tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@^2.1.0/dist/tfjs-backend-wasm.wasm');

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
