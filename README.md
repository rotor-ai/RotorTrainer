# Rotor Trainer

The RotorTrainer library supplies scripts for training various neural net models for use on an Android device. It is designed to be trained and implemented for the Rotor Vehicle ai agent, but can be used for any neural net classifier. All specific input requirements are listed in the descriptions of the models.

<img src="https://github.com/rotor-ai/RotorTrainer/blob/master/images/157_cam-image_array_.jpg">


<figure class="image">
    <img src="/images/157_cam-image_array_.jpg" alt="Example 128x128 jpg input image data">
    <figcaption>Example 128x128 jpg input image data</figcaption>
</figure>

### Prerequisites

The rotor trainer library uses the following dependencies among others:

- Miniconda3 (Dependency management)
- PyTorch
- Tensorflow
- Numpy
- SciKitLearn

All necessary dependencies are stored in python_environment.yml

### Installing

Download the RotorTrainer library to your computer:

```
git clone https://github.com/rotor-ai/RotorTrainer.git
```

Create the anaconda environment from the .yml file:

```
conda env create -f python_environment.yml
```

All data used for training must be placed in the /data/* subdirectory, and specified prior to running training. Training can be run via the rotor_trainer.py script:

```
python rotor_trainer.py
```

## Authors

* **Robert Humphrey** - [email](mailto:rhumphr6@gmail.com)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details