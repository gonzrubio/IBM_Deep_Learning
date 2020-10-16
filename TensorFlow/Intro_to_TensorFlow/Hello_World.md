<a href="https://www.skills.network/"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DL0120ENedX/labs/Template%20for%20Instructional%20Hands-on%20Labs/images/IDSNlogo.png" width="400px" align="center"></a>

<h1 align="center"><font size="5">TENSORFLOW'S HELLO WORLD</font></h1>


<h2>TENSORFLOW'S HELLO WORLD</h2>

<h3>Objective for this Notebook<h3>    
<h5> 1. How does TensorFlow work?</h5>
<h5> 2. Building a Graph. </h5>
<h5> 3. Meaning of Tensor? </h5>
<h5> 4. Defining multidimensional arrays using TensorFlow. </h5>
<h5> 5. How TensorFlow handles Variables. </h5>
<h5> 6. What are these Placeholders and what do they do? </h5>
<h5> 7. Learn Operations using TensorFlow. </h5>     


<div class="alert alert-block alert-info" style="margin-top: 20px">
<font size = 3><strong>In this notebook we will overview the basics of TensorFlow, learn it's structure and see what is the motivation to use it</strong></font>
<br>
<h2>Table of Contents</h2>
<ol>
    <li><a href="#ref2">How does TensorFlow work?</a></li>
    <li><a href="#ref3">tf.function and AutoGraph</a></li>
    <li><a href="#ref4">Defining multidimensional arrays using TensorFlow</a></li>
    <li><a href="#ref5">Why Tensors?</a></li>
    <li><a href="#ref6">Variables</a></li>
    <li><a href="#ref7">Operations</a></li>
</ol>
<p></p>
</div>
<br>

<hr>


<a id="ref2"></a>

<h2>How does TensorFlow work?</h2>

TensorFlow defines computations as Graphs, and these are made with operations (also know as “ops”). So, when we work with TensorFlow, it is the same as defining a series of operations in a Graph.

For example, the image below represents a graph in TensorFlow. _W_, _x_ and _b_ are tensors over the edges of this graph. _MatMul_ is an operation over the tensors_W_ and _b_, after that _Add_ is called and add the result of the previous operator with _b_. The resultant tensors of each operation cross the next one until the end where it's possible to get the wanted result.


<img src='https://ibm.box.com/shared/static/a94cgezzwbkrq02jzfjjljrcaozu5s2q.png'>


With TensorFlow 2.x, **Eager Execution** is enabled by default. This allows TensorFlow code to be executed and evaluated line by line. Before version 2.x was released, every graph had to be run wihthin a TensorFlow session. This only allowed for the entire graph to be run all at once. This would make debugging the computation graph each time. 


<h2>Installing TensorFlow </h2>

We begin by installing TensorFlow version 2.2.0 and its required prerequistes. 



```python
!pip install grpcio==1.24.3
!pip install tensorflow==2.2.0
```

    Collecting grpcio==1.24.3
    [?25l  Downloading https://files.pythonhosted.org/packages/30/54/c9810421e41ec0bca2228c6f06b1b1189b196b69533cbcac9f71b44727f8/grpcio-1.24.3-cp36-cp36m-manylinux2010_x86_64.whl (2.2MB)
    [K     |████████████████████████████████| 2.2MB 4.1MB/s eta 0:00:01     |██████▎                         | 430kB 4.1MB/s eta 0:00:01     |█████████████████▉              | 1.2MB 4.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: six>=1.5.2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from grpcio==1.24.3) (1.15.0)
    Installing collected packages: grpcio
      Found existing installation: grpcio 1.31.0
        Uninstalling grpcio-1.31.0:
          Successfully uninstalled grpcio-1.31.0
    Successfully installed grpcio-1.24.3
    Collecting tensorflow==2.2.0
    [?25l  Downloading https://files.pythonhosted.org/packages/3d/be/679ce5254a8c8d07470efb4a4c00345fae91f766e64f1c2aece8796d7218/tensorflow-2.2.0-cp36-cp36m-manylinux2010_x86_64.whl (516.2MB)
    [K     |████████████████████████████████| 516.2MB 35kB/s  eta 0:00:015  |▍                               | 5.8MB 4.7MB/s eta 0:01:49     |█                               | 14.5MB 6.8MB/s eta 0:01:15     |█                               | 17.3MB 6.8MB/s eta 0:01:14     |█▊                              | 28.1MB 5.9MB/s eta 0:01:24     |██▋                             | 42.9MB 7.1MB/s eta 0:01:08| 50.8MB 6.9MB/s eta 0:01:08     |████                            | 64.3MB 6.7MB/s eta 0:01:07     |██████                          | 96.9MB 7.1MB/s eta 0:01:00           | 100.8MB 6.8MB/s eta 0:01:02                  | 135.4MB 7.0MB/s eta 0:00:55     |█████████▌                      | 153.3MB 5.4MB/s eta 0:01:08��████▏                     | 163.6MB 43.2MB/s eta 0:00:09     |███████████                     | 176.3MB 5.9MB/s eta 0:00:58     |█████████████▎                  | 213.8MB 41.8MB/s eta 0:00:08     |██████████████▊                 | 238.0MB 36.0MB/s eta 0:00:08     |████████████████▏               | 261.7MB 39.6MB/s eta 0:00:07/s eta 0:00:07     |█████████████████               | 275.6MB 40.7MB/s eta 0:00:06     |█████████████████▏              | 276.3MB 40.7MB/s eta 0:00:06     |█████████████████▏              | 277.0MB 40.7MB/s eta 0:00:06     |█████████████████▋              | 283.2MB 38.3MB/s eta 0:00:07     |█████████████████▊              | 286.0MB 38.3MB/s eta 0:00:07     |█████████████████▉              | 288.1MB 38.3MB/s eta 0:00:06��██████████████▉             | 303.8MB 42.0MB/s eta 0:00:06     |███████████████████▊            | 318.8MB 5.5MB/s eta 0:00:37     | 329.2MB 5.4MB/s eta 0:00:35     | 332.7MB 5.4MB/s eta 0:00:34�█████████▋          | 349.5MB 6.9MB/s eta 0:00:25��█████████████████████▌         | 362.3MB 5.5MB/s eta 0:00:29��█████████████████████▌         | 363.6MB 5.5MB/s eta 0:00:28     |███████████████████████▏        | 373.4MB 6.9MB/s eta 0:00:212MB 6.9MB/s eta 0:00:21     |███████████████████████▋        | 381.0MB 5.5MB/s eta 0:00:25     |████████████████████████        | 387.5MB 5.5MB/s eta 0:00:24��█████████▌   | 459.4MB 5.5MB/s eta 0:00:11     |████████████████████████████▊   | 463.3MB 5.5MB/s eta 0:00:10     |█████████████████████████████▎  | 472.0MB 7.0MB/s eta 0:00:07     |█████████████████████████████▍  | 474.0MB 7.0MB/s eta 0:00:07     |█████████████████████████████▍  | 474.7MB 7.0MB/s eta 0:00:06     |█████████████████████████████▋  | 478.0MB 5.8MB/s eta 0:00:07��████████████████████▋  | 478.5MB 5.8MB/s eta 0:00:07     |█████████████████████████████▊  | 480.3MB 5.8MB/s eta 0:00:07     |█████████████████████████████▉  | 480.8MB 5.8MB/s eta 0:00:07     |█████████████████████████████▉  | 481.7MB 5.8MB/s eta 0:00:06    |██████████████████████████████▋ | 493.4MB 5.4MB/s eta 0:00:056.9MB 6.7MB/s eta 0:00:02     |███████████████████████████████▌| 508.9MB 5.5MB/s eta 0:00:02
    [?25hCollecting google-pasta>=0.1.8 (from tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/a3/de/c648ef6835192e6e2cc03f40b19eeda4382c49b5bafb43d88b931c4c74ac/google_pasta-0.2.0-py3-none-any.whl (57kB)
    [K     |████████████████████████████████| 61kB 21.7MB/s eta 0:00:01
    [?25hRequirement already satisfied: protobuf>=3.8.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorflow==2.2.0) (3.13.0)
    Collecting tensorflow-estimator<2.3.0,>=2.2.0 (from tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/a4/f5/926ae53d6a226ec0fda5208e0e581cffed895ccc89e36ba76a8e60895b78/tensorflow_estimator-2.2.0-py2.py3-none-any.whl (454kB)
    [K     |████████████████████████████████| 460kB 41.1MB/s eta 0:00:01
    [?25hCollecting wrapt>=1.11.1 (from tensorflow==2.2.0)
      Using cached https://files.pythonhosted.org/packages/82/f7/e43cefbe88c5fd371f4cf0cf5eb3feccd07515af9fd6cf7dbf1d1793a797/wrapt-1.12.1.tar.gz
    Requirement already satisfied: six>=1.12.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorflow==2.2.0) (1.15.0)
    Collecting scipy==1.4.1; python_version >= "3" (from tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/dc/29/162476fd44203116e7980cfbd9352eef9db37c49445d1fec35509022f6aa/scipy-1.4.1-cp36-cp36m-manylinux1_x86_64.whl (26.1MB)
    [K     |████████████████████████████████| 26.1MB 7.7MB/s eta 0:00:01ta 0:00:02
    [?25hRequirement already satisfied: absl-py>=0.7.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorflow==2.2.0) (0.10.0)
    Requirement already satisfied: termcolor>=1.1.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorflow==2.2.0) (1.1.0)
    Collecting gast==0.3.3 (from tensorflow==2.2.0)
      Downloading https://files.pythonhosted.org/packages/d6/84/759f5dd23fec8ba71952d97bcc7e2c9d7d63bdc582421f3cd4be845f0c98/gast-0.3.3-py2.py3-none-any.whl
    Requirement already satisfied: grpcio>=1.8.6 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorflow==2.2.0) (1.24.3)
    Collecting h5py<2.11.0,>=2.10.0 (from tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/60/06/cafdd44889200e5438b897388f3075b52a8ef01f28a17366d91de0fa2d05/h5py-2.10.0-cp36-cp36m-manylinux1_x86_64.whl (2.9MB)
    [K     |████████████████████████████████| 2.9MB 33.0MB/s eta 0:00:01   |███████████████████████████▉    | 2.5MB 33.0MB/s eta 0:00:01
    [?25hCollecting tensorboard<2.3.0,>=2.2.0 (from tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/1d/74/0a6fcb206dcc72a6da9a62dd81784bfdbff5fedb099982861dc2219014fb/tensorboard-2.2.2-py3-none-any.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 34.7MB/s eta 0:00:01
    [?25hCollecting keras-preprocessing>=1.1.0 (from tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/79/4c/7c3275a01e12ef9368a892926ab932b33bb13d55794881e3573482b378a7/Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42kB)
    [K     |████████████████████████████████| 51kB 21.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: numpy<2.0,>=1.16.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorflow==2.2.0) (1.19.1)
    Collecting opt-einsum>=2.3.2 (from tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/bc/19/404708a7e54ad2798907210462fd950c3442ea51acc8790f3da48d2bee8b/opt_einsum-3.3.0-py3-none-any.whl (65kB)
    [K     |████████████████████████████████| 71kB 4.8MB/s  eta 0:00:01
    [?25hRequirement already satisfied: wheel>=0.26; python_version >= "3" in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorflow==2.2.0) (0.35.1)
    Collecting astunparse==1.6.3 (from tensorflow==2.2.0)
      Downloading https://files.pythonhosted.org/packages/2b/03/13dde6512ad7b4557eb792fbcf0c653af6076b81e5941d36ec61f7ce6028/astunparse-1.6.3-py2.py3-none-any.whl
    Requirement already satisfied: setuptools in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from protobuf>=3.8.0->tensorflow==2.2.0) (49.6.0.post20200917)
    Requirement already satisfied: werkzeug>=0.11.15 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (1.0.1)
    Requirement already satisfied: requests<3,>=2.21.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2.24.0)
    Collecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0)
      Downloading https://files.pythonhosted.org/packages/7b/b8/88def36e74bee9fce511c9519571f4e485e890093ab7442284f4ffaef60b/google_auth_oauthlib-0.4.1-py2.py3-none-any.whl
    Collecting google-auth<2,>=1.6.3 (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/1f/cf/724b6436967a8be879c8de16b09fd80e0e7b0bcad462f5c09ee021605785/google_auth-1.22.1-py2.py3-none-any.whl (114kB)
    [K     |████████████████████████████████| 122kB 38.5MB/s eta 0:00:01
    [?25hCollecting tensorboard-plugin-wit>=1.6.0 (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/b6/85/5c5ac0a8c5efdfab916e9c6bc18963f6a6996a8a1e19ec4ad8c9ac9c623c/tensorboard_plugin_wit-1.7.0-py3-none-any.whl (779kB)
    [K     |████████████████████████████████| 788kB 9.6MB/s eta 0:00:01
    [?25hRequirement already satisfied: markdown>=2.6.8 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (3.2.2)
    Requirement already satisfied: idna<3,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2020.6.20)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (1.25.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (3.0.4)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0)
      Using cached https://files.pythonhosted.org/packages/a3/12/b92740d845ab62ea4edf04d2f4164d82532b5a0b03836d4d4e71c6f3d379/requests_oauthlib-1.3.0-py2.py3-none-any.whl
    Collecting cachetools<5.0,>=2.0.0 (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0)
      Using cached https://files.pythonhosted.org/packages/cd/5c/f3aa86b6d5482f3051b433c7616668a9b96fbe49a622210e2c9781938a5c/cachetools-4.1.1-py3-none-any.whl
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (0.2.8)
    Collecting rsa<5,>=3.1.4; python_version >= "3.5" (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0)
      Using cached https://files.pythonhosted.org/packages/1c/df/c3587a667d6b308fadc90b99e8bc8774788d033efcc70f4ecaae7fad144b/rsa-4.6-py3-none-any.whl
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2.0.0)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/05/57/ce2e7a8fa7c0afb54a0581b14a65b56e62b5759dbc98e80627142b8a3704/oauthlib-3.1.0-py2.py3-none-any.whl (147kB)
    [K     |████████████████████████████████| 153kB 36.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (0.4.8)
    Requirement already satisfied: zipp>=0.5 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (3.2.0)
    Building wheels for collected packages: wrapt
      Building wheel for wrapt (setup.py) ... [?25ldone
    [?25h  Stored in directory: /home/jupyterlab/.cache/pip/wheels/b1/c2/ed/d62208260edbd3fa7156545c00ef966f45f2063d0a84f8208a
    Successfully built wrapt
    Installing collected packages: google-pasta, tensorflow-estimator, wrapt, scipy, gast, h5py, oauthlib, requests-oauthlib, cachetools, rsa, google-auth, google-auth-oauthlib, tensorboard-plugin-wit, tensorboard, keras-preprocessing, opt-einsum, astunparse, tensorflow
      Found existing installation: scipy 1.5.2
        Uninstalling scipy-1.5.2:
          Successfully uninstalled scipy-1.5.2
      Found existing installation: gast 0.4.0
        Uninstalling gast-0.4.0:
          Successfully uninstalled gast-0.4.0
      Found existing installation: h5py 2.8.0
        Uninstalling h5py-2.8.0:
          Successfully uninstalled h5py-2.8.0
      Found existing installation: tensorboard 1.8.0
        Uninstalling tensorboard-1.8.0:
          Successfully uninstalled tensorboard-1.8.0
      Found existing installation: tensorflow 1.8.0
        Uninstalling tensorflow-1.8.0:
          Successfully uninstalled tensorflow-1.8.0
    Successfully installed astunparse-1.6.3 cachetools-4.1.1 gast-0.3.3 google-auth-1.22.1 google-auth-oauthlib-0.4.1 google-pasta-0.2.0 h5py-2.10.0 keras-preprocessing-1.1.2 oauthlib-3.1.0 opt-einsum-3.3.0 requests-oauthlib-1.3.0 rsa-4.6 scipy-1.4.1 tensorboard-2.2.2 tensorboard-plugin-wit-1.7.0 tensorflow-2.2.0 tensorflow-estimator-2.2.0 wrapt-1.12.1


**Restart kernel for latest version of TensorFlow to be activated**


<h2>Importing TensorFlow</h2>
<p>To use TensorFlow, we need to import the library. We imported it and optionally gave it the name "tf", so the modules can be accessed by <b>tf.module-name</b>:</p>



```python
import tensorflow as tf
if not tf.__version__ == '2.2.0':
    print(tf.__version__)
    raise ValueError('please upgrade to TensorFlow 2.2.0, or restart your Kernel (Kernel->Restart & Clear Output)')
```

IMPORTANT! => Please restart the kernel by clicking on "Kernel"->"Restart and Clear Outout" and wait until all output disapears. Then your changes are beeing picked up


* * *


<a id="ref3"></a>

# tf.function and AutoGraph


Now we call the TensorFlow functions that construct new <b>tf.Operation</b> and <b>tf.Tensor</b> objects. As mentioned, each <b>tf.Operation</b> is a <b>node</b> and each <b>tf.Tensor</b> is an edge in the graph.

Lets add 2 constants to our graph. For example, calling tf.constant([2], name = 'constant_a') adds a single <b>tf.Operation</b> to the default graph. This operation produces the value 2, and returns a <b>tf.Tensor</b> that represents the value of the constant.  
<b>Notice:</b> tf.constant([2], name="constant_a") creates a new tf.Operation named "constant_a" and returns a tf.Tensor named "constant_a:0".



```python
a = tf.constant([2], name = 'constant_a')
b = tf.constant([3], name = 'constant_b')
```

Lets look at the tensor **a**.



```python
a 
```




    <tf.Tensor: shape=(1,), dtype=int32, numpy=array([2], dtype=int32)>




```python
%whos
```

    Variable   Type           Data/Info
    -----------------------------------
    a          EagerTensor    tf.Tensor([2], shape=(1,), dtype=int32)
    b          EagerTensor    tf.Tensor([3], shape=(1,), dtype=int32)
    tf         module         <module 'tensorflow' from<...>/tensorflow/__init__.py'>


As you can see, it just shows the name, shape and type of the tensor in the graph. We will see it's value by running the TensorFlow code as shown below.



```python
tf.print(a.numpy()[0])
```

    2


Annotating the python functions with **tf.function** uses TensorFlow Autograph to create a TensorFlow static execution graph for the function.   tf.function annotation tells TensorFlow Autograph to transform function _add_ into TensorFlow control flow, which then defines the TensorFlow static execution graph. 



```python
@tf.function
def add(a,b):
    c = tf.add(a, b)
    #c = a + b is also a way to define the sum of the terms
    print(c)
    return c

```


```python
result = add(a,b)
tf.print(result[0])
```

    Tensor("Add:0", shape=(1,), dtype=int32)
    5


Even this silly example of adding 2 constants to reach a simple result defines the basis of TensorFlow. Define your operations (In this case our constants and _tf.add_), define a Python function named _add_ and decorate it with using the _tf.function_ annotator.


<h3>What is the meaning of Tensor?</h3>

<div class="alert alert-success alertsuccess" style="margin-top: 20px">
<font size = 3><strong>In TensorFlow all data is passed between operations in a computation graph, and these are passed in the form of Tensors, hence the name of TensorFlow.</strong></font>
<br>
<br>
    The word <b>tensor</b> from new latin means "that which stretches". It is a mathematical object that is named "tensor" because an early application of tensors was the study of materials stretching under tension. The contemporary meaning of tensors can be taken as multidimensional arrays. 

</div>

That's great, but... what are these multidimensional arrays? 

Going back a little bit to physics to understand the concept of dimensions:<br>
<img src="https://ibm.box.com/shared/static/ymn0hl3hf8s3xb4k15v22y5vmuodnue1.svg"/>

<div style="text-align:center"><a href="https://en.wikipedia.org/wiki/Dimension">Image Source</a></div>
<br>

The zero dimension can be seen as a point, a single object or a single item.

The first dimension can be seen as a line, a one-dimensional array can be seen as numbers along this line, or as points along the line. One dimension can contain infinite zero dimension/points elements.

The second dimension can be seen as a surface, a two-dimensional array can be seen as an infinite series of lines along an infinite line. 

The third dimension can be seen as volume, a three-dimensional array can be seen as an infinite series of surfaces along an infinite line.

The Fourth dimension can be seen as the hyperspace or spacetime, a volume varying through time, or an infinite series of volumes along an infinite line. And so forth on...


As mathematical objects: <br><br>
<img src="https://ibm.box.com/shared/static/kmxz570uai8eeg6i6ynqdz6kmlx1m422.png">

<div style="text-align: center"><a href="https://book.mql4.com/variables/arrays">Image Source</a></div>


Summarizing:<br><br>

<table style="width:100%">
  <tr>
    <td><b>Dimension</b></td>
    <td><b>Physical Representation</b></td> 
    <td><b>Mathematical Object</b></td>
    <td><b>In Code</b></td>
  </tr>
  
  <tr>
    <td>Zero </td>
    <td>Point</td> 
    <td>Scalar (Single Number)</td>
    <td>[ 1 ]</td>
  </tr>

  <tr>
    <td>One</td>
    <td>Line</td> 
    <td>Vector (Series of Numbers) </td>
    <td>[ 1,2,3,4,... ]</td>
  </tr>
  
   <tr>
    <td>Two</td>
    <td>Surface</td> 
    <td>Matrix (Table of Numbers)</td>
       <td>[ [1,2,3,4,...], [1,2,3,4,...], [1,2,3,4,...],... ]</td>
  </tr>
  
   <tr>
    <td>Three</td>
    <td>Volume</td> 
    <td>Tensor (Cube of Numbers)</td>
    <td>[ [[1,2,...], [1,2,...], [1,2,...],...], [[1,2,...], [1,2,...], [1,2,...],...], [[1,2,...], [1,2,...], [1,2,...] ,...]... ]</td>
  </tr>
  
</table>


* * *


<a id="ref4"></a>

<h2>Defining multidimensional arrays using TensorFlow</h2>
Now we will try to define such arrays using TensorFlow:



```python

Scalar = tf.constant(2)
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )

print ("Scalar (1 entry):\n %s \n" % Scalar)

print ("Vector (3 entries) :\n %s \n" % Vector)

print ("Matrix (3x3 entries):\n %s \n" % Matrix)

print ("Tensor (3x3x3 entries) :\n %s \n" % Tensor)
```

    Scalar (1 entry):
     tf.Tensor(2, shape=(), dtype=int32) 
    
    Vector (3 entries) :
     tf.Tensor([5 6 2], shape=(3,), dtype=int32) 
    
    Matrix (3x3 entries):
     tf.Tensor(
    [[1 2 3]
     [2 3 4]
     [3 4 5]], shape=(3, 3), dtype=int32) 
    
    Tensor (3x3x3 entries) :
     tf.Tensor(
    [[[ 1  2  3]
      [ 2  3  4]
      [ 3  4  5]]
    
     [[ 4  5  6]
      [ 5  6  7]
      [ 6  7  8]]
    
     [[ 7  8  9]
      [ 8  9 10]
      [ 9 10 11]]], shape=(3, 3, 3), dtype=int32) 
    


<b>tf.shape</b> returns the shape of our data structure.



```python
Scalar.shape
```




    TensorShape([])




```python
Tensor.shape
```




    TensorShape([3, 3, 3])



Now that you understand these data structures, I encourage you to play with them using some previous functions to see how they will behave, according to their structure types:



```python

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

@tf.function
def add():
    add_1_operation = tf.add(Matrix_one, Matrix_two)
    return add_1_operation



print ("Defined using tensorflow function :")
add_1_operation = add()
print(add_1_operation)
print ("Defined using normal expressions :")
add_2_operation = Matrix_one + Matrix_two
print(add_2_operation)
```

    Defined using tensorflow function :
    tf.Tensor(
    [[3 4 5]
     [4 5 6]
     [5 6 7]], shape=(3, 3), dtype=int32)
    Defined using normal expressions :
    tf.Tensor(
    [[3 4 5]
     [4 5 6]
     [5 6 7]], shape=(3, 3), dtype=int32)


With the regular symbol definition and also the TensorFlow function we were able to get an element-wise multiplication, also known as Hadamard product. <br>

But what if we want the regular matrix product?

We then need to use another TensorFlow function called <b>tf.matmul()<b>:



```python

Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

@tf.function
def mathmul():
  return tf.matmul(Matrix_one, Matrix_two)


mul_operation = mathmul()

print ("Defined using tensorflow function :")
print(mul_operation)


```

    Defined using tensorflow function :
    tf.Tensor(
    [[13 18]
     [18 25]], shape=(2, 2), dtype=int32)


We could also define this multiplication ourselves, but there is a function that already does that, so no need to reinvent the wheel!


* * *


<a id="ref5"></a>

<h2>Why Tensors?</h2>

The Tensor structure helps us by giving the freedom to shape the dataset in the way we want.

And it is particularly helpful when dealing with images, due to the nature of how information in images are encoded,

Thinking about images, its easy to understand that it has a height and width, so it would make sense to represent the information contained in it with a two dimensional structure (a matrix)... until you remember that images have colors, and to add information about the colors, we need another dimension, and thats when Tensors become particularly helpful.

Images are encoded into color channels, the image data is represented into each color intensity in a color channel at a given point, the most common one being RGB, which means Red, Blue and Green. The information contained into an image is the intensity of each channel color into the width and height of the image, just like this:

<img src='https://ibm.box.com/shared/static/xlpv9h5xws248c09k1rlx7cer69y4grh.png'>
<a href="https://msdn.microsoft.com/en-us/library/windows/desktop/dn424131.aspx">Image Source</a>

So the intensity of the red channel at each point with width and height can be represented into a matrix, the same goes for the blue and green channels, so we end up having three matrices, and when these are combined they form a tensor. 


* * *


<a id="ref6"></a>

# Variables

Now that we are more familiar with the structure of data, we will take a look at how TensorFlow handles variables.
<b>First of all, having tensors, why do we need variables?</b>  
TensorFlow variables are used to share and persist some stats that are manipulated by our program. That is, when you define a variable, TensorFlow adds a <b>tf.Operation</b> to your graph. Then, this operation will store a writable tensor value. So, you can update the value of a variable through each run.


Let's first create a simple counter, by first initializing a variable _v_ that will be increased one unit at a time:



```python
v = tf.Variable(0)
```

We now create a python method _increment_by_one_. This method will internally call _td.add_ that takes in two arguments, the <b>reference_variable</b> to update, and assign it to the <b>value_to_update</b> it by.



```python
@tf.function
def increment_by_one(v):
        v = tf.add(v,1)
        return v
```

To update the value of the variable _v_, we simply call the _increment_by_one_ method and pass the variable to it. We will invoke this method thrice. This method will increment the variable by one and print the updated value each time. 



```python
for i in range(3):
    v = increment_by_one(v)
    print(v)
```

    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)


* * *


<a id="ref7"></a>

<h2>Operations</h2>

Operations are nodes that represent the mathematical operations over the tensors on a graph. These operations can be any kind of functions, like add and subtract tensor or maybe an activation function.

<b>tf.constant</b>, <b>tf.matmul</b>, <b>tf.add</b>, <b>tf.nn.sigmoid</b> are some of the operations in TensorFlow. These are like functions in python but operate directly over tensors and each one does a specific thing. 

<div class="alert alert-success alertsuccess" style="margin-top: 20px">Other operations can be easily found in: <a href="https://www.tensorflow.org/versions/r0.9/api_docs/python/index.html">https://www.tensorflow.org/versions/r0.9/api_docs/python/index.html</a></div>



```python
a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)


print ('c =: %s' % c)
    
print ('d =: %s' % d)
```

    c =: tf.Tensor([7], shape=(1,), dtype=int32)
    d =: tf.Tensor([3], shape=(1,), dtype=int32)


<b>tf.nn.sigmoid</b> is an activation function, it's a little more complicated, but this function helps learning models to evaluate what kind of information is good or not.


* * *


## Want to learn more?

Running deep learning programs usually needs a high performance platform. **PowerAI** speeds up deep learning and AI. Built on IBM’s Power Systems, **PowerAI** is a scalable software platform that accelerates deep learning and AI with blazing performance for individual users or enterprises. The **PowerAI** platform supports popular machine learning libraries and dependencies including TensorFlow, Caffe, Torch, and Theano. You can use [PowerAI on IMB Cloud](https://cocl.us/ML0120EN_PAI).

Also, you can use **Watson Studio** to run these notebooks faster with bigger datasets.**Watson Studio** is IBM’s leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, **Watson Studio** enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of **Watson Studio** users today with a free account at [Watson Studio](https://cocl.us/ML0120EN_DSX).This is the end of this lesson. Thank you for reading this notebook, and good luck on your studies.


### Thanks for completing this lesson!

Notebook created by: <a href="https://linkedin.com/in/romeo-kienzler-089b4557"> Romeo Kienzler </a>, <a href="https://linkedin.com/in/saeedaghabozorgi"> Saeed Aghabozorgi </a> and <a href="https://ca.linkedin.com/in/rafaelblsilva"> Rafael Belo Da Silva </a>

Updated to TF 2.X by  <a href="https://www.linkedin.com/in/samaya-madhavan"> Samaya Madhavan </a>


### References:


## Change Log

| Date (YYYY-MM-DD) | Version | Changed By | Change Description                                          |
| ----------------- | ------- | ---------- | ----------------------------------------------------------- |
| 2020-09-21        | 2.0     | Srishti    | Migrated Lab to Markdown and added to course repo in GitLab |

<hr>

## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>


[https://www.tensorflow.org/versions/r0.9/get_started/index.html](https://www.tensorflow.org/versions/r0.9/get_started/index.html?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)<br>
[http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html](http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)<br>
[https://www.tensorflow.org/versions/r0.9/api_docs/python/index.html](https://www.tensorflow.org/versions/r0.9/api_docs/python/index.html?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)<br>
<a href="https://www.tensorflow.org/api_docs/python/">https://www.tensorflow.org/versions/r0.9/resources/dims_types.html</a><br>
[https://en.wikipedia.org/wiki/Dimension]\([https://en.wikipedia.org/wiki/Dimension?cm_mmc=Email_Newsletter-](https://en.wikipedia.org/wiki/Dimension?cm_mmc=Email_Newsletter-&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)<br>
[https://book.mql4.com/variables/arrays]\([https://book.mql4.com/variables/arrays?cm_mmc=Email_Newsletter-](https://book.mql4.com/variables/arrays?cm_mmc=Email_Newsletter-&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)<br>
[https://msdn.microsoft.com/en-us/library/windows/desktop/dn424131(v=vs.85).aspx]\([https://msdn.microsoft.com/en-us/library/windows/desktop/dn424131(v=vs.85).aspx?cm_mmc=Email_Newsletter-](https://msdn.microsoft.com/en-us/library/windows/desktop/dn424131(v=vs.85).aspx?cm_mmc=Email_Newsletter-&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)<br>


<hr>

Copyright © 2018 [Cognitive Class](https://cocl.us/DX0108EN_CC). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ).

