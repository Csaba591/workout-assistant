# Workout Assistant
## Explore the UCF101 dataset
1. Open Dataset.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EQrePHpLFadgS7Zl1FigE1YdU_vOZJNX)
2. Run all
3. Change cells to see other videos

## Exercise classifier training
1. Open Training.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N9_Eoh1dAaiXhQjIYMDTU1q8_SoG92eX)
2. Select a GPU environment
3. Train and validate
    1. Run all until model.evaluate()
4. Test
    1. Upload your test images/videos
    2. Change the filenames in the corresponding cells
    3. Test the model with your chosen material
    
## Run web app
1. Clone the repository
2. Open a terminal in the repo directory
    1. Install [node.js and npm](https://nodejs.org/)
    2. Install http-server: `npm install --global http-server`
    3. `cd views/`
    4. Run `http-server --port 5500`
3. Open localhost:5500 in a browser
