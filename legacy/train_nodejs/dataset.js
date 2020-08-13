const glob = require( 'glob' )
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { parse, serialize } = require('tfjs-npy');
const npyjs = require('npyjs');

const {faceNetConfig, extractFaceFeature} = require('./model/facenet')
const {poseNetConfig, extractPoseFeature} = require('./model/posenet')


//This class is adapted from https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/controller_dataset.js
/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
class ControllerDataset {
  constructor() {
      this.xsFace = null;
      this.xsPose = null;
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
  async addDir(directory, labels) {
    console.log("addDir start");
    for (let labelIndex = 0; labelIndex < labels.length; labelIndex ++){
        let label = labels[labelIndex];
        const labelDir = path.join(directory, label);
        let image_paths = glob.sync(labelDir + "/*.jpg")
        image_paths = image_paths.concat(glob.sync(labelDir + "/*.png"))
        for (let i = 0; i < image_paths.length; i ++){
            let image_path = image_paths[i];
            console.log(image_path);
            const buf = fs.readFileSync(image_path);
            
            const image = tf.node.decodeJpeg(buf, 3);
            const faceFeature = await extractFaceFeature(image);
            const poseFeature = await extractPoseFeature(image);
            if (faceFeature == null || poseFeature == null){
                continue;
            }
            // One-hot encode the label.
            const y = tf.tidy(
                () => tf.cast(tf.oneHot(tf.tensor1d([labelIndex]).toInt(), labels.length)), 'float32');

            if (this.xsFace == null || this.xsPose == null) {
                // For the first example that gets added, keep example and y so that the
                // ControllerDataset owns the memory of the inputs. This makes sure that
                // if addExample() is called in a tf.tidy(), these Tensors will not get
                // disposed.
                this.xsFace = tf.keep(faceFeature);
                this.xsPose = tf.keep(poseFeature);
                this.ys = tf.keep(y);
            } else {
                const oldXsFace = this.xsFace;
                const oldXsPose = this.xsPose;
                this.xsFace = tf.keep(oldXsFace.concat(faceFeature, 0));
                this.xsPose = tf.keep(oldXsPose.concat(poseFeature, 0));

                const oldY = this.ys;
                this.ys = tf.keep(oldY.concat(y, 0));

                oldXsFace.dispose();
                oldXsPose.dispose()
                oldY.dispose();
                y.dispose();
            }
        }
    };
    console.log("addDir end")
    };

    async loadFromNpy(
        xsFaceNpy="./saved_data/train_face.npy", 
        xsPoseNpy="./saved_data/train_pose.npy", 
        ysNpy="./saved_data/train_y_float32.npy") {
        [this.xsFace, this.xsPose, this.ys] = await Promise.allSettled(
            [loadNpy(xsFaceNpy), loadNpy(xsPoseNpy), loadNpy(ysNpy)]);
        this.xsFace = this.xsFace.value.clone();
        this.xsPose = this.xsPose.value.clone();
        this.ys = this.ys.value.clone();
        console.log(this.xsFace);
    }
};

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

    // let n = new npyjs();
    // n.load(fn, (array, shape) => {
    //     // `array` is a one-dimensional array of the raw data
    //     // `shape` is a one-dimensional array that holds a numpy-style shape.
    //     console.log(
    //         `You loaded an array with ${array.length} elements and ${shape.length} dimensions.`
    //     );
    // });
    // return array
}

exports.ControllerDataset = ControllerDataset;
exports.loadNpy = loadNpy;

const labels = ["true", "false"];


// (async () => {
//     let trainDataset = new ControllerDataset(labels);
//     await trainDataset.addDir("../samples/afewva/0.5", labels);
//     await trainDataset.addDir("../samples/affwild1/train/0.1", labels);
//     await trainDataset.addDir("../samples/affwild2/train/0.1", labels);
//     saveNpy(trainDataset.xsFace, "./train_face.npy");
//     saveNpy(trainDataset.xsPose, "./train_pose.npy");
//     saveNpy(trainDataset.ys, "./train_y.npy");
//   })();