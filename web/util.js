const fs = require('fs');
const path = require('path');
const { Canvas, Image, ImageData} = require('canvas');
const { JSDOM } = require('jsdom');

// Write Javascript object to the same path as the input file.
function writeJson(jsonObj, frameFilePath) {
    jsonContent = JSON.stringify(jsonObj);
    jsonPath = path.format({
        dir: path.dirname(frameFilePath),
        name: path.basename(frameFilePath, path.extname(frameFilePath)),
        ext: '.json'
    });
    fs.writeFile(jsonPath, jsonContent, 'utf8', function (err) {
        if (err) {
            console.log("An error occured while writing JSON Object to File.");
            return console.log(err);
        }
    
        console.log("JSON file has been saved.");
    });
}

// Load opencv.js just like before but using Promise instead of callbacks:
// This function is adapted from opencv js official tutorial.
function loadOpenCV() {
    return new Promise(resolve => {
      global.Module = {
        onRuntimeInitialized: resolve
      };
      global.cv = require('./opencv.js');
    });
}

// Using jsdom and node-canvas we define some global variables to emulate HTML DOM.
// Although a complete emulation can be archived, here we only define those globals used
// by cv.imread() and cv.imshow().
// This function is adapted from opencv js official tutorial.
function installDOM() {
    const dom = new JSDOM();
    global.document = dom.window.document;
    // The rest enables DOM image and canvas and is provided by node-canvas
    global.Image = Image;
    global.HTMLCanvasElement = Canvas;
    global.ImageData = ImageData;
    global.HTMLImageElement = Image;
    }

exports.writeJson = writeJson;
exports.loadOpenCV = loadOpenCV;
exports.installDOM = installDOM;