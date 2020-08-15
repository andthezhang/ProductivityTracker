import * as tf from '@tensorflow/tfjs';

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
  const model = await tf.loadGraphModel('http://localhost:8080/no_pose_affectnet_model_tfjs/model.json');
  return model;
}
async function extractFeature(image, truncatedMobileNet, verbose=true){
  const resizeImage = tf.image.resizeBilinear(image.expandDims(0), [224, 224]);
  const xs = await truncatedMobileNet.predict(resizeImage);
  return xs;
}

exports.extractFaceFeature = extractFaceFeature;
// exports.extractPoseFeature = extractPoseFeature;
exports.loadTruncatedMobileNet = loadTruncatedMobileNet;
exports.loadCustomModel = loadCustomModel;
exports.extractFeature = extractFeature;