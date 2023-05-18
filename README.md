# IUM Project

## Authors
*Kacper Klassa*  
*Jakub Kowieski*

## Repository structure

* **`artifacts`** - contains all the artifacts created during data preprocessing and modeling  
  * **`models`** - contains the trained models  
  * **`preprocessing`** - contains the data preprocessing artifacts  
* **`data`** - contains data used in training and testing (excluded from version control)
* **`notebooks`** - contains the jupyter notebooks used for data preprocessing, modeling and testing
  * [`data_analysys.ipynb`](./notebooks/data_analysys.ipynb) - exploratory data analysis (stage I of the project)
  * [`data_preparation.ipynb`](./notebooks/data_preperation.ipynb) - data augmentation and export to CSV
  * [`model_training.ipynb`](./notebooks/model_training.ipynb) - data preprocessing, model training and evaluation
  * [`inference_testing.ipynb`](./notebooks/inference_testing.ipynb) - loading and testing the trained models, exploring programmatic model serving
* **`scripts`** - contains test and utility scripts
* **`services`** - contains the code and configuration files for model serving and load balancer services
  * **`load-balancer-svc`** - contains the Nginx load balancer service configuration files
  * **`torch-model-svc`** - contains the PyTorch model serving code and configuration files
  * **`xgboost-model-svc`** - contains the XGBoost model serving code and configuration files


## Running the project

Go the `services` directory and run the following command:

```
docker compose up
```

The API gateway will be available at [`http://localhost:8080/`](http://localhost:8080/). The API documentation is available at [`http://localhost:8080/docs`](http://localhost:8080/docs).  

The Nginx reverse proxy performs round robin load balancing, so the request will get forwarded by the proxy to a randomly selected upstream service. If you want to access a specific service, you can find them at [`http://localhost:8080/torch-model/`](http://localhost:8080/torch-model/) and [`http://localhost:8080/xgb-model/`](http://localhost:8080/xgb-model/) for the Neural Network and XGBoost models respectively. You can also bypass the proxy and access the model serving services directly at [`http://localhost:8081/`](http://localhost:8081/) and [`http://localhost:8082/`](http://localhost:8082/).
