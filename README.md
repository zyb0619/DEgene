
### Set up 
安装所需要包的命令如下：
```
pip install -r requirements.txt
pip install torch_cluster-latest+cu102-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-latest+cu102-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-latest+cu102-cp37-cp37m-linux_x86_64.whl
pip install scanpy
```

### Prepare dataset
在使用splatter模拟好数据后，将其放入该项目的目录下（此处已经放入，为sim_data.h5ad），然后运行[simulated_data.ipynb.py](/simulated_data.ipynb.py)即可得到graphSVX能够使用的文件。


### To train a model 
如果要训练一个GCN网络，可以运行[script_train.py](/simulated_train.py): 
```
# python3 simulated_train.py --save=true --dataset='simulated' --model='GCN' --multiclass=True
```

### To predict cell type
如果要使用GCN来预测空间转录组数据中每个细胞的类型，则可以运行[pred.ipynb](/pred.ipynb)

### To explain a model using GraphSVX
如果要使用graphSVX寻找marker gene，则运行[simulated_explain.py](/simulated_explain.py):
```
python3 simulated_explain.py --dataset='sim1' --model='GCN' --info=True
```

### Compare with Wilcoxon sum rank
要与wilcoxon秩和方法做比较，并评估graphSVX的效果则需要运行[evaluate.ipynb](/evaluate.ipynb)
