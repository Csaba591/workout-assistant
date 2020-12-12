const timeChartElem = document.getElementById("time-chart");
const exerciseChartElem = document.getElementById("exercise-chart");

const summaryHolder = document.getElementById("summary-holder");
const startHolder = document.getElementById("start-holder");
const cameraOnHolder = document.getElementById("camera-on-holder");

const workoutTimeElem = document.getElementById("workout-time");
const exerciseNameElem = document.getElementById("exercise-name");
const pushUpNumElem = document.getElementById("push-up-num");
const squatNumElem = document.getElementById("squat-num");
const pullUpNumElem = document.getElementById("pull-up-num");

const startsInHolder = document.getElementById("starts-in-holder");
const startsInElem = document.getElementById("starts-in")
const startedHolder = document.getElementById("started-holder")

const startButton = document.getElementById("btn-start");
const finishButton = document.getElementById("btn-finish");
const restartButton = document.getElementById("btn-restart");

let pushUpNum = 0;
let pullUpNum = 0;
let squatNum = 0;

let workoutTime = 0;
let pushUpTime = 0;
let pullUpTime = 0;
let squatTime = 0;
let unknownTime = 0;

let workoutList = [];

const workoutTypes = {
    PUSHUP: 0,
    PULLUP: 1,
    SQUAT: 2,
    UNKNOWN: 3
};
const typesArray = ['Push up', 'Pull up', 'Squat', 'Unknown']

startButton.onclick = () => {
    startHolder.classList.add("gone");
    cameraOnHolder.classList.remove("gone");
    enableCam();
    let startsInCount = 3;
    const startsInInterval = setInterval(() => {
        startsInElem.innerText = startsInCount;
        startsInCount--;
    }, 1000);

    setTimeout(() => {
        clearInterval(startsInInterval);
        startsInHolder.classList.add('gone');
        startedHolder.classList.remove('gone');
        startTimer()
    }, 4000);
};

finishButton.onclick = () => {
    cameraOnHolder.classList.add("gone");
    summaryHolder.classList.remove("gone");
    drawDiagrams();
};

restartButton.onclick = () => {
    summaryHolder.classList.add("gone");
    cameraOnHolder.classList.remove("gone");
    resetCounters();
    let startsInCount = 3;
    const startsInInterval = setInterval(() => {
        startsInElem.innerText = startsInCount;
        startsInCount--;
    }, 1000);

    setTimeout(() => {
        clearInterval(startsInInterval);
        startsInHolder.classList.add('gone');
        startedHolder.classList.remove('gone');
        startTimer()
    }, 3000);
};

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

function addToWorkoutList(type) {
    if (workoutList.length >= 10)
        workoutList.pop();
    workoutList.unshift(type);
}
function getCurrentTypeIndex() {
    const numOfTypes = [];
    numOfTypes.push(workoutList.filter(type => type == workoutTypes.PUSHUP).length);
    numOfTypes.push(workoutList.filter(type => type == workoutTypes.PULLUP).length);
    numOfTypes.push(workoutList.filter(type => type == workoutTypes.SQUAT).length);
    numOfTypes.push(workoutList.filter(type => type == workoutTypes.UNKNOWN).length);

    return numOfTypes.indexOf(Math.max(...numOfTypes));
}
function increaseCurrentTypeNum(index) {
    switch (index) {
        case workoutTypes.PUSHUP:
            pushUpNum++;
            break;
        case workoutTypes.PULLUP:
            pullUpNum++;
            break;
        case workoutTypes.SQUAT:
            squatNum++
            break;
    }
}
function increaseCurrentTypeTimer(index) {
    switch (index) {
        case workoutTypes.PUSHUP:
            pushUpTime++;
            break;
        case workoutTypes.PULLUP:
            pullUpTime++;
            break;
        case workoutTypes.SQUAT:
            squatTime++;
            break;
        default:
            unknownTime++;
            break;
    }
}
function syncDomElements(index) {
    pushUpNumElem.innerText = pushUpNum;
    pullUpNumElem.innerText = pullUpNum;
    squatNumElem.innerText = squatNum;

    exerciseNameElem.innerText = typesArray[index];
}

let secondInterval
function startTimer() {
    secondInterval = setInterval(() => {
        workoutTime++;
        workoutTimeElem.innerText = workoutTime;
        increaseCurrentTypeTimer(getCurrentTypeIndex())
    }, 1000);
}

function resetCounters() {
    pullUpNum = 0;
    pushUpNum = 0;
    squatNum = 0;
    workoutTime = 0;
    pushUpTime = 0;
    pullUpTime = 0;
    squatTime = 0;
    unknownTime = 0;
    workoutList = [];
    exerciseName = "Unkown";
}
let timeChart
let exerciseChart

function drawDiagrams() {
    const backgroundColor = [
        'rgba(255, 99, 132, 0.2)',
        'rgba(54, 162, 235, 0.2)',
        'rgba(255, 206, 86, 0.2)',
        'rgba(75, 192, 192, 0.2)'
    ];
    const borderColor = [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)'
    ];
    
    const timeChartData = {
        datasets: [{
            data: [pushUpTime, pullUpTime, squatTime, unknownTime],
            backgroundColor: backgroundColor,
            borderColor: borderColor,
            borderWidth: 1
        }],
        
        labels: [
            'Push up time',
            'Pull up time',
            'Squat time',
            'Unknown time'
        ]
    };
    
    const exerciseChartData = {
        datasets: [{
            data: [pushUpNum, pullUpNum, squatNum],
            backgroundColor: backgroundColor,
            borderColor: borderColor,
            borderWidth: 1
        }],
        
        labels: [
            'Push up count',
            'Pull up count',
            'Squat count'
        ]
    };
    const options = {
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero: true
                }
            }]
        }
    }
    
    timeChart = new Chart(timeChartElem, {
        type: 'pie',
        data: timeChartData,
        options: options
    });
    
    exerciseChart = new Chart(exerciseChartElem, {
        type: 'pie',
        data: exerciseChartData,
        options: options
    });    
}

const video = document.getElementById('webcam');
const videoout = document.getElementById('videoout');
const percent = document.getElementById('percent');

let classifier;
let poseNet;

tf.loadLayersModel("http://localhost:5500/mobilenet-js_model/v2/model.json")
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
    startButton.disabled = false;
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
    startButton.disabled = false;
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
let lastDir = false;
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
  let type;
  if (certainty < 0.9)
    type = workoutTypes.UNKNOWN;
  else {
    let index = await predsTensor.argMax(1).data();
    index = index[0];
    switch (index) {
        case 0:
            type = workoutTypes.SQUAT;
            break
        case 1:
            type = workoutTypes.PULLUP;
            break
        case 2:
            type = workoutTypes.PUSHUP;
            break
    }
  }
    addToWorkoutList(type);
    syncDomElements(getCurrentTypeIndex());
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
    

    draw(position)
    if (ys.length == 5)
      ys.shift()
    ys.push(position.y)
    const rising = isRising(ys);
    if (!rising && lastDir) {
      increaseCurrentTypeNum(getCurrentTypeIndex())
    }
    lastDir = rising;
    window.requestAnimationFrame(predictPose);
  });
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