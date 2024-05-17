# Cliamte StyleGAN-Tensorflow
**Simple & Intuitive** Tensorflow implementation of *"A Style-Based Generator Architecture for Generative Adversarial Networks"* **[NeurIPS 2020 Workshop poster](https://ai4earthscience.github.io/neurips-2020-workshop/)**


<div align="center">
  <img src=./assets/stylegan-teaser.png>
</div>

### [Official code](https://github.com/NVlabs/stylegan) | [Paper](https://arxiv.org/abs/1812.04948) | [Video](https://www.youtube.com/watch?v=kSLJriaOumA&feature=youtu.be) | [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset) 






### To create seven channel dataset for climate data :
``` 
python src/untar_all_six_variables.py
python src/six_variables_climateset.py
python src/combine_seven_channels_numpy.py
```
The cropped data will be located in folder : ```"/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_seven_channels_by_year/"``` with the following channel order : 

["pr", "prw", "psl", "ts", "ua850", "va850", "omega"]



### Train
```
> python main.py --dataset FFHQ --img_size 1024 --gpu_num 4 --progressive True --phase train
```

### Test
```
> python main.py --dataset FFHQ --img_size 1024 --progressive True --batch_size 16 --phase test
```

### Draw
#### Figure02 uncurated
```
python main.py --dataset FFHQ --img_size 1024 --progressive True --phase draw --draw uncurated
```

#### Figure03 style mixing
```
python main.py --dataset FFHQ --img_size 1024 --progressive True --phase draw --draw style_mix
```

#### Figure08 truncation trick
```
CUDA_VISIBLE_DEVICES=1 bash  ./new_bash_scripts/main.sh 
```

## Architecture
<div align="center">
  <img src=./assets/A_module.png>
  <img src=./assets/B_module.png>
</div>

## Our Results (1024x1024)
* Training time: **2 days 14 hours** with **V100 * 4**
### Uncurated
<div align="center">
  <img src=./assets/uncurated.jpg>
</div>

### Style mixing
<div align="center">
  <img src=./assets/style_mix_glod_bold.png>
</div>

### Truncation trick
<div align="center">
  <img src=./assets/truncation_trick.png>
</div>

### Generator loss graph
<div align="center">
  <img src="./assets/g_loss.png">
</div>

### Discriminator loss graph
<div align="center">
  <img src="./assets/d_loss.png">
</div>

## Author
[Rishabh Gupta](http://linedin.com/in/rishabh-gupta-ai/)
