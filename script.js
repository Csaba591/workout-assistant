const counter = document.getElementById('counter');
const enableButton = document.getElementById('enable');
const video = document.getElementById('webcam');
const videoout = document.getElementById('videoout');
const state = document.getElementById('state');
const percent = document.getElementById('percent');

let OpenCVReady = false;
let cap;
let currentFrame;
let lastFrame;
let green
function OpenCVLoaded() {
  cv['onRuntimeInitialized'] = () => {
    console.log('OpenCV loaded!');
    OpenCVReady = true;
    cap = new cv.VideoCapture(video);
    currentFrame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    lastFrame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    green = new cv.Scalar(0, 255, 0);
  };
}

let modelReady = false;
function warmUpModel() {
  console.log('Making warmup prediction...');
  const dummyInput = tf.ones([1, 224, 224, 3]);
  const offset = tf.scalar(127.5);
  dummyInput.toFloat()
  .sub(offset)
  .div(offset)
  .expandDims();
  const dummyPreds = classifier.predict(dummyInput);
  dummyPreds.dispose();
  dummyInput.dispose();
  modelReady = true;
  console.log('Done.');
}

let classifier = null;
tf.loadLayersModel("http://localhost:8080/mobilenet-js_model/v2/model.json")
.then((model) => {
  console.log('Classifier loaded!');
  classifier = model;
  warmUpModel();
});

let poseNet = undefined;
posenet.load({
  architecture: 'MobileNetV1',
  outputStride: 16,
  inputResolution: { width: 400, height: 320 },
  multiplier: 0.75,
  quantBytes: 2
}).then(function(model) {
  console.log('PoseNet loaded!');
  poseNet = model;
});

window.setInterval(() => {
  if (typeof poseNet !== 'undefined' && modelReady && OpenCVReady)
    enableButton.disabled = false;
}, 500);

function enableCam() {
  if (typeof poseNet === 'undefined' || !modelReady || !OpenCVReady)
    return;

  console.log('Camera on');
  const constraints = { video: true };
  navigator.mediaDevices.getUserMedia(constraints)
  .then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictExercise);
    //video.addEventListener('loadeddata', count);
    video.addEventListener('loadeddata', predictPose);
  });
}

let lastGray;
function count() {
  cap.read(currentFrame);
  
  const imgGray = new cv.Mat(video.height, video.width, cv.CV_8UC1);
  cv.cvtColor(currentFrame, imgGray, cv.COLOR_RGBA2GRAY);

  if (typeof lastGray === 'undefined') {
    lastGray = imgGray;
    window.requestAnimationFrame(count);
    return;
  }

  const imgDiff = new cv.Mat(video.height, video.width, cv.CV_8UC1);
  cv.absdiff(lastGray, imgGray, imgDiff);
  lastGray.delete();
  lastGray = imgGray;
  
  /*const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5,5));
  const imgDilated = new cv.Mat(video.height, video.width, cv.CV_8UC1);
  console.log(imgDiff.type())
  cv.dilate(imgDiff, imgDilated, new cv.Point(-1, 1), 4);
  kernel.delete();*/
  
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(imgDiff, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
  cv.drawContours (imgDiff, contours, -1, green, 10, cv.LINE_8, hierarchy, 1);

  

  // show
  videoout.width = video.width
  videoout.height = video.height
  cv.imshow('videoout', imgDiff);
  //imgDilated.delete();
  imgDiff.delete();
  
  window.requestAnimationFrame(count);
}

let ys = [];
const maxReps = 5;
let reps = 0;
let lastDir = false;

function resetCounter() {
  counter.innerText = 0;
  reps = 0;
}

const classes = ['BodyWeightSquats', 'PullUps', 'PushUps']
const predictedLabels = [];
async function predictExercise() {
  tf.engine().startScope();
  const offset = tf.scalar(127.5);
  const imgTensor = tf.browser.fromPixels(video)
  .resizeBilinear([224, 224])
  .toFloat()
  .sub(offset)
  .div(offset)
  .expandDims();
  const predsTensor = classifier.predict(imgTensor);
  let certainty = await predsTensor.max().data();
  certainty = certainty[0];
  percent.innerText = Math.round(certainty*100) + '%';
  let label;
  if (certainty < 0.9)
    label = 'Unknown';
  else {
    let index = await predsTensor.argMax(1).data();
    index = index[0];
    label = classes[index];
  }
  state.innerText = label;
  predictedLabels.push(label);
  if (predictedLabels.length > 5)
    predictedLabels.shift();
  tf.engine().endScope();
  window.requestAnimationFrame(predictExercise);
}

function draw(position) {
  videoout.width = video.width;
  videoout.height = video.height;
  const ctx = videoout.getContext('2d');
  ctx.drawImage(video, 0, 0, video.width, video.height);
  ctx.fillStyle = 'green';
  ctx.beginPath();
  ctx.arc(position.x, position.y, 5, 0, 2 * Math.PI);
  ctx.fill();
}

const keypointIndices = [0, 1, 2, 3, 4, 5, 6];
function predictPose() {
  poseNet.estimateSinglePose(video, {
    flipHorizontal: false
  }).then(pose => {
    if (reps >= maxReps) {
      window.requestAnimationFrame(predictPose);
      return;
    }

    // calculate upper body y position
    keypoints = pose.keypoints.slice(
      keypointIndices[0], 
      keypointIndices[keypointIndices.length-1]
    )
    let xPos = 0;
    let div = 0;
    keypoints.forEach(kp => {
      if (kp.score > 0.85) {
        xPos += kp.position.x
        div++;
      }
    });
    xPos /= div;
    let yPos = 0;
    keypoints.forEach(kp => {
      if (kp.score > 0.85)
        yPos += kp.position.y
    });
    yPos /= div;
    position = { x: xPos, y: yPos };
    
    console.log(keypoints);

    draw(position)
    if (ys.length == 5)
      ys.shift()
    ys.push(position.y)
    const rising = isRising(ys);
    if (!rising && lastDir) {
      reps++;
      counter.innerText = reps;
      saveImage(videoout.toDataURL());
    }
    lastDir = rising;
    window.requestAnimationFrame(predictPose);
  });
}

function saveImage(dataUrl) {
  const img = document.createElement('img');
  img.src = dataUrl;
  document.body.appendChild(img);
}

const eps = video.height / 10;
function isRising(ys) {
  if (ys.length <= 1)
    return false
  baseline = ys[0]
  total = 0.0
  for (let i = 1; i < ys.length; i++)
    total += ys[i] - baseline
  if (Math.abs(total) < eps)
    return undefined
  return total < 0;
}