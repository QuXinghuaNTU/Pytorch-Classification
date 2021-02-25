# Pytorch-Classification
This is an example code for classification using pytorch

In general, you can go to the main.py to run the code:
```bash run.sh ``` or
```python main.py --input_dim 2 --layer_num 2 --hidden_layer_dims [20,20] --out_dim 2 --max_epoch 1000 --learning_rate 0.0001```


## Detail settings are shown below
### model creation:
```from model import Intellik_Net
model = Intellik_Net(input_dim, layer_num, hidden_layer_dim_list, out_dim```
In model.py, 
```Intellik_Net(input_dim, layer_num, hidden_layer_dim_list, out_dim) automitically create the NN model for the training based on user's hyperparameter settings.```

## parameter settings: 
### input_dim ---> dimension of the observations collected from IntelliK

### out_dim ---> dimension of the actions collected from IntelliK 

### layer_num  ---> how many hidden layers you want for your network

### hidden_layer_dim_list ---> The list that show the dimension of your each hidden dimension


## Data loader:
load the data from the dolder ~/training_data/: 
```from data_loader import DataLoad
data_dir = "{}".format(your_data_dir)
name = "{}".format(name_of_dataset)
loader = DataLoad(data_dir, name)```



