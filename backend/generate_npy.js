/**
 * NodeJS script to generate npy files for dataset.
 */
const glob = require( 'glob' )
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const blazeface = require('@tensorflow-models/blazeface');
const { parse, serialize } = require('tfjs-npy');


//This class is adapted from https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/controller_dataset.js
/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
class ControllerDataset {
  constructor() {
      this.xsFace = null;
      this.xs = null;
      this.ys = null;
  }

    /**
    * Load dataset from directory. Each sub-directory corresponding to one class.
    * For example, 
    *      directory = ./dataset
    *      subDirNames = ["true", "false"]
    * Fetch all images in ./dataset/true with label "true"
    * Fetch all images in ./dataset/false with label "false"
    * 
    * @param {string} directory Parent directory of dataset.
    * @param {Array} labels Array of labels in string.
    */
  async addDir(directory, labels, truncatedMobileNet, blazefaceModel) {
    for (let labelIndex = 0; labelIndex < labels.length; labelIndex ++){
        let label = labels[labelIndex];
        const labelDir = path.join(directory, label);
        let image_paths = glob.sync(labelDir + "/*.jpg")
        image_paths = image_paths.concat(glob.sync(labelDir + "/*.png"))
        image_paths = image_paths.concat(glob.sync(labelDir + "/*.JPG"))
        for (let i = 0; i < image_paths.length; i ++){
            let image_path = image_paths[i];
            console.log(image_path);
            const buf = fs.readFileSync(image_path);
            
            const image = tf.node.decodeJpeg(buf, 3);
            const faceFeature = await extractFaceFeature(image, blazefaceModel, truncatedMobileNet);
            const resizeImage = tf.image.resizeBilinear(image.expandDims(0), [224, 224]);
            const xs = await truncatedMobileNet.predict(resizeImage);
            if (faceFeature == null || xs == null){
                continue;
            }
            // One-hot encode the label.
            const y = tf.tidy(
                () => tf.cast(tf.oneHot(tf.tensor1d([labelIndex]).toInt(), labels.length), 'float32'));

            if (this.xsFace == null || this.xs == null) {
                // For the first example that gets added, keep example and y so that the
                // ControllerDataset owns the memory of the inputs. This makes sure that
                // if addExample() is called in a tf.tidy(), these Tensors will not get
                // disposed.
                this.xsFace = tf.keep(faceFeature);
                this.xs = tf.keep(xs);
                this.ys = tf.keep(y);
            } else {
                const oldXsFace = this.xsFace;
                const oldxs = this.xs;
                this.xsFace = tf.keep(oldXsFace.concat(faceFeature, 0));
                this.xs = tf.keep(oldxs.concat(xs, 0));

                const oldY = this.ys;
                this.ys = tf.keep(oldY.concat(y, 0));

                oldXsFace.dispose();
                oldxs.dispose()
                oldY.dispose();
                y.dispose();
            }
        }
    };
    };

    async loadFromNpy(xsFaceNpy, xsNpy, ysNpy) {
        [this.xsFace, this.xs, this.ys] = await Promise.allSettled(
            [loadNpy(xsFaceNpy), loadNpy(xsNpy), loadNpy(ysNpy)]);
        this.xsFace = this.xsFace.value.clone();
        this.xs = this.xs.value.clone();
        this.ys = this.ys.value.clone();
    }
};
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
async function extractFaceFeature(image, blazefaceModel, truncatedMobileNet, verbose=true){
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

async function saveNpy(t, outputFile){
    const serializeT = await serialize(t);
    fs.appendFileSync(outputFile, new Buffer(serializeT));
}

function bufferToArrayBuffer(b){
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

async function loadNpy(fn){
    const b = fs.readFileSync(fn);
    const ab = bufferToArrayBuffer(b);
    return await parse(ab);
}

const labels = ["false", "true"];


(async () => {
    const truncatedMobileNet = await loadTruncatedMobileNet();
    const blazefaceModel = await blazeface.load();
    let trainDataset = new ControllerDataset(labels);
    // await trainDataset.addDir("../samples/afewva/0.5", labels, truncatedMobileNet, blazefaceModel);
    await trainDataset.addDir("../samples/affwild1/train/0.1", labels, truncatedMobileNet, blazefaceModel);
    await trainDataset.addDir("../samples/affwild2/train/0.1", labels, truncatedMobileNet, blazefaceModel);
    await trainDataset.addDir("../samples/affectnet/train/0.1", labels, truncatedMobileNet, blazefaceModel);
    await trainDataset.addDir("../samples/affectnet/valid/0.1", labels, truncatedMobileNet, blazefaceModel);
    saveNpy(trainDataset.xsFace, "./saved_data/xs_face_no_pose.npy");
    saveNpy(trainDataset.xs, "./saved_data/xs_no_pose.npy");
    saveNpy(trainDataset.ys, "./saved_data/ys_no_pose.npy");
  })();
