
<p align="center">
<h1 align="center"> Continual Diffuser (CoD) </h1>
</p>



Code for "Continual Diffuser: Endowing Diffuser Plasticity and Stability with Experience Rehearsal in Continual Offline Reinforcement Learning".

![Framework](figures/framework.png)

First clone the code and installation of the relevant package.

    pip install -r requirements.txt

Before you start we strongly recommend that you register a `wandb` account.
This will record graphs and curves during the experiment.
If you want, complete the login operation in your shell. Enter the following command and follow the prompts to complete the login.

    wandb login

API keys can be found in User Settings page https://wandb.ai/settings. For more information you can refer to https://docs.wandb.ai/quickstart .

Next is how to replicate all experiments:
## For Model Training

If use default training config:

    python continual_diffuser_train.py 

The instruction of how to set the various hyperparameters will be updated soon.

