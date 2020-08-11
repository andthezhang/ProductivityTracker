const glob = require( 'glob' )
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

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
    for (let labelIndex = 0; labelIndex < labels.length; labelIndex ++){
        let label = labels[labelIndex];
        const labelDir = path.join(directory, label);
        const image_paths = glob.sync(labelDir + "/*.jpg")
        for (let i = 0; i < image_paths.length; i ++){
            let image_path = image_paths[i];
            const buf = fs.readFileSync(image_path);
            
            const image = tf.node.decodeJpeg(buf, 3);
            const faceFeature = await extractFaceFeature(image);
            const poseFeature = await extractPoseFeature(image);
            // One-hot encode the label.
            const y = tf.tidy(
                () => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), labels.length));

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
                console.log(this.xsFace);
                this.xsFace = tf.keep(tf.concat([oldXsFace, faceFeature], 0));
                this.xsPose = tf.keep(oldXsPose.concat(poseFeature, 0));

                const oldY = this.ys;
                this.ys = tf.keep(oldY.concat(y, 0));

                oldXsFace.dispose();
                oldXsPose.dispose()
                oldY.dispose();
                y.dispose();
            }
        }
    }};
};

exports.ControllerDataset = ControllerDataset;