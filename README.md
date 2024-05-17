# Cliamte StyleGAN-Tensorflow
**Simple & Intuitive** Tensorflow implementation of *"A Style-Based Generator Architecture for Generative Adversarial Networks"* **[NeurIPS 2020 Workshop poster](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_53.pdf)**



### [Prior Official code](https://github.com/NVlabs/stylegan) | [Paper](https://arxiv.org/abs/1812.04948)



### To create seven channel dataset for climate data :
``` 
python src/untar_all_six_variables.py
python src/six_variables_climateset.py
python src/combine_seven_channels_numpy.py
```
The cropped data will be located in folder : ```"/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_seven_channels_by_year/"``` with the following channel order : 

["pr", "prw", "psl", "ts", "ua850", "va850", "omega"]



### Train for annealed climate training 


```
CUDA_VISIBLE_DEVICES=1 bash  ./new_bash_scripts/climate_training_128_omega_wo_progressive_custom_cropping.sh
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

## Credits to 
[Junho Kim](https://bit.ly/junho-kim)
