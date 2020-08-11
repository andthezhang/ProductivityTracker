const tf = require('@tensorflow/tfjs-node');
const facemesh = require('@tensorflow-models/facemesh');
const blazeface = require('@tensorflow-models/blazeface');

const faceNetConfig ={
    outputBatchShape : [1, 7, 7, 256]
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
async function extractFaceFeature(image, verbose=false){
    // Extract image embedding from MobileNet.
    const blazefaceModel = await blazeface.load();
    const blazefacePredictions = await blazefaceModel.estimateFaces(image, {returnTensors: true});
  
    // Crop and Resize to Mobile net input size.
    const boxes = blazefacePredictions[0].topLeft.concat(blazefacePredictions[0].bottomRight);
    const croppedFace = tf.image.cropAndResize(image.expandDims(0), boxes.expandDims(0), [0], [224, 224]);
    truncatedMobileNet = await loadTruncatedMobileNet();
    const embeddings = truncatedMobileNet.predict(croppedFace);
    if (verbose){
      console.log("extractFaceFeature=======")
      console.log(embeddings.shape)
    }
    return embeddings;
}

exports.faceNetConfig = faceNetConfig;
exports.extractFaceFeature = extractFaceFeature;