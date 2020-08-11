const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');

const poseNetConfig ={
    outputBatchShape : [1, 22, 11]
}
// Extract Pose Feature.
async function extractPoseFeature(image, scoreThreshold=0.6, verbose=false){
    const upperPoseCount = 11;
    const maxDistance = Math.pow((image.shape[0]), 2) + Math.pow((image.shape[1]), 2);
    poseNet = await posenet.load({
      architecture: 'MobileNetV1',
      quantBytes: 1
    });
    const pose = await poseNet.estimateSinglePose(image, {
      flipHorizontal: true
    });
    upperPose = pose.keypoints.slice(0, upperPoseCount) // only getting coordindate of upper body.
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
    distanceKernelTensor = tf.tensor(distanceKernel);
    cosineKernelTensor = tf.tensor(cosineKernel);
  
    // Concat two kernels and return.
    returnTensor = tf.concat([distanceKernelTensor, cosineKernelTensor]).expandDims(0);
  
    if (verbose){
      console.log("extractPoseFeature=======")
      console.log(returnTensor.shape)
      // distanceKernelTensor.print();
      // cosineKernelTensor.print();
      // returnTensor.print()
    }
  
    return returnTensor;
}

exports.poseNetConfig = poseNetConfig
exports.extractPoseFeature = extractPoseFeature