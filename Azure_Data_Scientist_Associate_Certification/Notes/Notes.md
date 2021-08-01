---
title: "Azure Data Scientist Associate Certification Notes"
author: "Bruttocao, Marco (858067)^[858067@stud.unive.it]"
subtitle: Notes and Procedures
output:
  pdf_document:
    latex_engine: xelatex
    includes:
      in_header: ./preamble.txt
  html_document:
    df_print: paged
biblio-style: apalike
link-citations: yes
documentclass: article
capsize: normalsize
fontsize: 10pt
geometry: margin=1in
spacing: doublespacing
footerdate: yes
# abstract: null
# bibliography: null
toc: yes
editor_options: 
  markdown: 
    wrap: 72
---

\newpage

# Build and operate machine learning solutions with Azure Machine Learning

## Introduction to the Azure Machine Learning SDK

A workspace is a context for the experiments, data, compute targets, and other 
assets associated with a machine learning workload (Workspaces are Azure resources, 
and as such they are defined within a resource group in an Azure subscription). 
The assets in a workspace include:

* Compute targets for development, training, and deployment
* Data for experimentation and model training
* Notebooks containing shared code and documentation
* Experiments, including run history with logged metrics and outputs
* Pipelines that define orchestrated multi-step processes
* Models that you have trained

The Azure resources created alongside a workspace include:

* A storage account
* An Application Insights instance, used to monitor predictive services in the workspace
* An Azure Key Vault instance, used to manage secrets such as authentication keys and credentials used by the workspace
* A container registry, created as-needed to manage containers for deployed models

You can create a workspace in any of the following ways:

* In the Microsoft Azure portal, create a new Machine Learning resource, 
specifying the subscription, resource group and workspace name
* Use the Azure Machine Learning Python SDK to run code that creates a workspace
* Use the Azure Command Line Interface (CLI) with the Azure Machine Learning CLI extension
* Create an Azure Resource Manager template

Azure Machine Learning studio is a web-based tool for managing an Azure Machine 
Learning workspace. It enables you to create, manage, and view all of the assets 
in your workspace and provides the following graphical tools:

* Designer, a drag and drop interface for "no code" machine learning model development.
* Automated Machine Learning, a wizard interface that enables you to train a model 
using a combination of algorithms and data preprocessing techniques to find the best
model for your data.

While graphical interfaces like Azure Machine Learning studio make it easy to create 
and manage machine learning assets, it is often advantageous to use a code-based 
approach to managing resources. By writing scripts to create and manage resources, you can:

* Run machine learning operations from your preferred development environment
* Automate asset creation and configuration to make it repeatable
* Ensure consistency for resources that must be replicated in multiple environments
* Incorporate machine learning asset configuration into developer operations (DevOps) workflows, 
such as continuous integration / continuous deployment (CI/CD) pipelines

### Installing the Azure Machine Learning SDK for Python

Azure Machine Learning provides software development kits (SDKs) for Python and R, 
which you can use to create, manage, and use assets in an Azure Machine Learning workspace.
You can install the Azure Machine Learning SDK for Python by using the pip package 
management utility, as shown in the following code sample:

```python
pip install azureml-sdk
```

The SDK is installed using the Python pip utility, and consists of the main 
azureml-sdk package as well as numerous other ancillary packages that contain 
specialized functionality. For example, the azureml-widgets package provides 
support for interactive widgets in a Jupyter notebook environment. 
To install additional packages, include them in the pip install command:

```python
pip install azureml-sdk azureml-widgets
```
[SDK Documentation](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py)

After installing the SDK package in your Python environment, you can write code to 
connect to your workspace and perform machine learning operations. The easiest way 
to connect to a workspace is to use a workspace configuration file, which includes 
the Azure subscription, resource group, and workspace details as shown here:

```json
{
    "subscription_id": "1234567-abcde-890-fgh...",
    "resource_group": "aml-resources",
    "workspace_name": "aml-workspace"
}
```
> You can download a configuration file for a workspace from the Overview page 
of its blade in the Azure portal or from Azure Machine Learning studio

To connect to the workspace using the configuration file, you can use the 
from_config method of the Workspace class in the SDK, as shown here:

```python
from azureml.core import Workspace
ws = Workspace.from_config()
```
By default, the from_config method looks for a file named config.json in the folder 
containing the Python code file, but you can specify another path if necessary.

As an alternative to using a configuration file, you can use the get method of 
the Workspace class with explicitly specified subscription, resource group, 
and workspace details as shown here - though the configuration file technique 
is generally preferred due to its greater flexibility when using multiple scripts:

```python
from azureml.core import Workspace

ws = Workspace.get(name='aml-workspace',
                   subscription_id='1234567-abcde-890-fgh...',
                   resource_group='aml-resources')
```
Whichever technique you use, if there is no current active session with your 
Azure subscription, you will be prompted to authenticate.

The Workspace class is the starting point for most code operations. 
For example, you can use its compute_targets attribute to retrieve a dictionary
object containing the compute targets defined in the workspace, like this:

```python
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)
```

The SDK contains a rich library of classes that you can use to create, manage, 
and use many kinds of asset in an Azure Machine Learning workspace.

### The Azure Machine Learning CLI Extension

The Azure command-line interface (CLI) is a cross-platform command-line tool for 
managing Azure resources. The Azure Machine Learning CLI extension is an additional 
package that provides commands for working with Azure Machine Learning.

To install the Azure Machine Learning CLI extension, you must first install the Azure CLI.

### Compute Instances

Azure Machine Learning includes the ability to create Compute Instances in a workspace 
to provide a development environment that is managed with all of the other 
assets in the workspace.

Compute Instances include Jupyter Notebook and JupyterLab installations that you 
can use to write and run code that uses the Azure Machine Learning SDK to work 
with assets in your workspace.

You can choose a compute instance image that provides the compute specification you need, 
from small CPU-only VMs to large GPU-enabled workstations. Because compute instances 
are hosted in Azure, you only pay for the compute resources when they are running; 
so you can create a compute instance to suit your needs, and stop it when your 
workload has completed to minimize costs.

You can store notebooks independently in workspace storage, and open them in any 
compute instance.

## Azure Machine Learning experiments

In Azure Machine Learning, an experiment is a named process, usually the running 
of a script or a pipeline, that can generate metrics and outputs and be tracked 
in the Azure Machine Learning workspace. An experiment can be run multiple times, 
with different data, code, or settings; and Azure Machine Learning tracks each run, 
enabling you to view run history and compare results for each run. When you submit 
an experiment, you use its run context to initialize and end the experiment run 
that is tracked in Azure Machine Learning, as shown in the following code sample:

```python
from azureml.core import Experiment

# create an experiment variable
experiment = Experiment(workspace = ws, name = "my-experiment")

# start the experiment
run = experiment.start_logging()

# experiment code goes here

# end the experiment
run.complete()
```

After the experiment run has completed, you can view the details of the run in the 
Experiments tab in Azure Machine Learning studio. Experiments are most useful when 
they produce metrics and outputs that can be tracked across runs. In addition to 
logging metrics, an experiment can generate output files. Often these are trained 
machine learning models, but you can save any sort of file and make it available 
as an output of your experiment run. The output files of an experiment are saved 
in its outputs folder. You can upload local files to the run's outputs folder by 
using the Run object's upload_file method in your experiment code.

You can run an experiment inline using the start_logging method of the Experiment 
object, but it's more common to encapsulate the experiment logic in a script and 
run the script as an experiment. The script can be run in any valid compute context, 
making this a more flexible solution for running experiments as scale. An experiment 
script is just a Python code file that contains the code you want 
to run in the experiment. To access the experiment run context (which is needed 
to log metrics) the script must import the azureml.core.Run class and call its 
get_context method. 

```python
from azureml.core import Run
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
data = pd.read_csv('data.csv')

# Count the rows and log the result
row_count = (len(data))
run.log('observations', row_count)

# Save a sample of the data
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
```

## Train a machine learning model with Azure Machine Learning

You can use a ScriptRunConfig to run a script-based experiment that trains a machine 
learning model. When using an experiment to train a model, your script should save 
the trained model in the outputs folder.

```python
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Prepare the dataset
diabetes = pd.read_csv('data.csv')
X, y = diabetes[['Feature1','Feature2','Feature3']].values, diabetes['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()
```
To prepare for an experiment that trains a model, a script like this is created
and saved in a folder. For example, you could save this script as training_script.py 
in a folder named training_folder. Since the script includes code to load training 
data from data.csv, this file should also be saved in the folder.

```python
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create a Python environment for the experiment
sklearn_env = Environment("sklearn-env")

# Ensure the required packages are installed
packages = CondaDependencies.create(conda_packages=['scikit-learn','pip'],
                                    pip_packages=['azureml-defaults'])
sklearn_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory='training_folder',
                                script='training.py',
                                environment=sklearn_env) 

# Submit the experiment
experiment = Experiment(workspace=ws, name='training-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion()
```

To pass parameter values to a script being run in an experiment, you need to 
provide an arguments value containing a list of comma-separated arguments and 
their values to the ScriptRunConfig:

```python
# Create a script config
script_config = ScriptRunConfig(source_directory='training_folder',
                                script='training.py',
                                arguments = ['--reg-rate', 0.1],
                                environment=sklearn_env)
```

After an experiment run has completed, you can use the run objects get_file_names 
method to list the files generated. Standard practice is for scripts that train
models to save them in the run's outputs folder. You can also use the run object's 
download_file and download_files methods to  download output files to the local file system.

Model registration enables you to track multiple versions of a model, and retrieve models 
for inferencing (predicting label values from new data). When you register a model, you can 
specify a name, description, tags, framework (such as Scikit-Learn or PyTorch), framework 
version, custom properties, and other useful metadata. Registering a model with the same 
name as an existing model automatically creates a new version of the model, 
starting with 1 and increasing in units of 1.

## Work with Data in Azure Machine Learning

In Azure Machine Learning, datastores are abstractions for cloud data sources. 
They encapsulate the information required to connect to data sources. 
You can access datastores directly in code by using the Azure Machine Learning SDK, 
and use it to upload or download data. Azure Machine Learning supports the creation of 
datastores for multiple kinds of Azure data source, including:

* Azure Storage (blob and file containers)
* Azure Data Lake stores
* Azure SQL Database
* Azure Databricks file system (DBFS)

Every workspace has two built-in datastores (an Azure Storage blob container, 
and an Azure Storage file container) that are used as system storage by Azure Machine Learning.
To add a datastore to your workspace, you can register it using the graphical 
interface in Azure Machine Learning studio, or you can use the Azure Machine Learning SDK.

```python
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()

# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(workspace=ws, 
                                                  datastore_name='blob_data', 
                                                  container_name='data_container',
                                                  account_name='az_store_acct',
                                                  account_key='123456abcde789.')
```

Datasets are versioned packaged data objects that can be easily consumed in 
experiments and pipelines. Datasets are the recommended way to work with data, 
and are the primary mechanism for advanced Azure Machine Learning capabilities 
like data labeling and data drift monitoring. You can use the visual interface in 
Azure Machine Learning studio or the Azure Machine Learning SDK to create datasets 
from individual files or multiple file paths.

```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
csv_paths = [(blob_ds, 'data/files/current_data.csv'),
             (blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
```

To create a file dataset using the SDK, use the from_files method of the Dataset.File class:

```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')
```
After registering a dataset, you can retrieve it by using any of the following techniques:

* The datasets dictionary attribute of a Workspace object.
* The get_by_name or get_by_id method of the Dataset class.

```python
import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['csv_table']

# Get a dataset by name from the datasets class
ds2 = Dataset.get_by_name(ws, 'img_files')
```
Datasets can be versioned, enabling you to track historical versions of datasets 
that were used in experiments, and reproduce those experiments with data in the same state.
You can create a new version of a dataset by registering it with the same name as 
a previously registered dataset and specifying the create_new_version property:

```python
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
             (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
```

You can retrieve a specific version of a dataset by specifying the version parameter 
in the get_by_name method of the Dataset class.

```python
img_ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)
```

When you need to access a dataset in an experiment script, you must pass the 
dataset to the script. There are two ways you can do this. You can pass a tabular 
dataset as a script argument. When you take this approach, the argument received 
by the script is the unique ID for the dataset in your workspace. In the script, 
you can then get the workspace from the run context and use it to retrieve the 
dataset by it's ID.

```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds],
                                environment=env) 
```

```python
from azureml.core import Run, Dataset

parser.add_argument('--ds', type=str, dest='dataset_id')
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace
dataset = Dataset.get_by_id(ws, id=args.dataset_id)
data = dataset.to_pandas_dataframe()
```

Alternatively, you can pass a tabular dataset as a named input. In this approach, 
you use the as_named_input method of the dataset to specify a name for the dataset. 
Then in the script, you can retrieve the dataset by name from the run context's 
input_datasets collection without needing to retrieve it from the workspace. 
Note that if you use this approach, you still need to include a script argument 
for the dataset, even though you don't actually use it to retrieve the dataset.


```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds.as_named_input('my_dataset')],
                                environment=env)
```

```python
from azureml.core import Run

parser.add_argument('--ds', type=str, dest='ds_id')
args = parser.parse_args()

run = Run.get_context()
dataset = run.input_datasets['my_dataset']
data = dataset.to_pandas_dataframe()
```

You can pass a file dataset as a script argument. Unlike with a tabular dataset, 
you must specify a mode for the file dataset argument, which can be as_download or 
as_mount. This provides an access point that the script can use to read the files 
in the dataset. In most cases, you should use as_download, which copies the files 
to a temporary location on the compute where the script is being run. However, 
if you are working with a large amount of data for which there may not be enough 
storage space on the experiment compute, use as_mount to stream the files directly 
from their source.

You can also pass a file dataset as a named input. In this approach, you use the 
as_named_input method of the dataset to specify a name before specifying the access 
mode. Then in the script, you can retrieve the dataset by name from the run context's 
input_datasets collection and read the files from there. As with tabular datasets, 
if you use a named input, you still need to include a script argument for the dataset, 
even though you don't actually use it to retrieve the dataset.

## Work with Compute in Azure Machine Learning












