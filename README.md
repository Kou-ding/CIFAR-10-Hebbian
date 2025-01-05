# CIFAR-10-Hebbian
A Hebbian Learning inspired implementation of a neural network trained on the CIFAR-10 dataset. This is the third and final assignment of the course "Neural Networks - Deep Learning".

### Python pyenv
To control different versions of python you can use pyenv:
```bash
# Add this to .bashrc
if command -v pyenv 1>/dev/null 2>&1; then
   eval "$(pyenv init -)" 
fi

# Install python version using 
pyenv install 3.12.0 # for example 3.12.0

# Enable this version
pyenv shell 3.12.0

# Now you are using this version of python
# You can check using
python --version

# This is a folder persistent way 
pyenv local 3.12.0
```