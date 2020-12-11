const counter = document.getElementById('counter');
const enableButton = document.getElementById('enable');
const video = document.getElementById('webcam');
const videoout = document.getElementById('videoout');
const state = document.getElementById('state');
const percent = document.getElementById('percent');
const repImages = document.getElementById('reps');

let classifier;
let poseNet;

tf.loadLayersModel("http://localhost:8080/mobilenet-js_model/v2/model.json")
.then((model) => {
  console.log('Classifier loaded!');
  classifier = model;
  warmUpModel();
});

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
  if (typeof poseNet !== 'undefined')
    enableButton.disabled = false;
}

posenet.load({
  architecture: 'MobileNetV1',
  outputStride: 16,
  inputResolution: { width: 400, height: 320 },
  multiplier: 0.75,
  quantBytes: 2
}).then(function(model) {
  console.log('PoseNet loaded!');
  poseNet = model;
  if (modelReady)
    enableButton.disabled = false;
});

function enableCam() {
  if (typeof poseNet === 'undefined' || !modelReady)
    return;

  console.log('Camera on');
  const constraints = { video: true };
  navigator.mediaDevices.getUserMedia(constraints)
  .then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictExercise);
    video.addEventListener('loadeddata', predictPose);
  });
}

let ys = [];
const maxReps = 5;
let reps = 0;
let lastDir = false;

function resetCounter() {
  counter.innerText = 0;
  reps = 0;
  while (repImages.hasChildNodes())
    repImages.removeChild(repImages.lastChild);
}

const classes = ['Squats', 'PullUps', 'PushUps'];
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

const firstIndex = 0;
const lastIndex = 6;
function predictPose() {
  poseNet.estimateSinglePose(video, {
    flipHorizontal: false
  }).then(pose => {
    if (reps >= maxReps) {
      window.requestAnimationFrame(predictPose);
      return;
    }

    // calculate upper body y position
    keypoints = pose.keypoints.slice(firstIndex, lastIndex+1);
    let xPos = 0;
    let div = 0;
    keypoints.forEach(kp => {
      if (kp.score >= 0.85) {
        xPos += kp.position.x
        div++;
      }
    });
    xPos /= div;
    let yPos = 0;
    keypoints.forEach(kp => {
      if (kp.score >= 0.85)
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
  repImages.appendChild(img);
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