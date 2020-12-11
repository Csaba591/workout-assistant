const counter = document.getElementById('counter');
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
classifier = tf.loadLayersModel("http://localhost:8080/mobilenet-js_model/v2/model.json")
.then((model) => {
  console.log('Classifier loaded!');
  classifier = model;
  warmUpModel();
});

function enableCam() {
  if (!(modelReady && OpenCVReady))
    return;

  console.log('Camera');
  const constraints = { video: true };
  navigator.mediaDevices.getUserMedia(constraints)
  .then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictExercise);
    video.addEventListener('loadeddata', count);
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
  /*
  if (typeof lastGray === 'undefined') {
    lastGray = currentGray.clone();
    currentGray.delete();
    console.log('Got lastGray');
    window.requestAnimationFrame(count);
    return;
  }
  const kernel = cv.getStructuringElement(cv.MORPH_RECT,new cv.Size(5,5));
  const diff = new cv.Mat();
  cv.absdiff(currentGray, lastGray, diff);*/
  //cv.dilate(diff, diff, kernel, iterations=4);
  
  /*let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(dst, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
  cv.drawContours (dst, contours, -1, color, 1, cv.LINE_8, hierarchy, 10)
  
  lastGray = currentGray.clone();*/
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

const keypointIndices = [0, 1, 2, 3, 4, 5, 6];
function predictPose() {
  poseNet.estimateSinglePose(video, {
    flipHorizontal: false
  }).then((pose) => {
    if (reps >= maxReps)
      return;

    // nose y position
    keypoints = pose.keypoints.slice(
      keypointIndices[0], 
      keypointIndices[keypointIndices.length-1]
    )
    let xPos = 0;
    keypoints.forEach(kp => xPos += kp.position.x);
    xPos /= keypointIndices.length;
    let yPos = 0;
    keypoints.forEach(kp => yPos += kp.position.y);
    yPos /= keypointIndices.length;
    position = { x: xPos, y: yPos };
    console.log(xPos + ' ' + yPos);
    //drawAll(positions)
    draw(position)
    if (ys.length == 2)
      ys.shift()
    ys.push(position.y)
    const rising = isRising(ys);
    state.innerText = (pose.keypoints[5].score + pose.keypoints[5].score) / 2;
    if (!rising && lastDir) {
      reps++;
      counter.innerText = reps;
      saveImage(canvas.toDataURL());
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