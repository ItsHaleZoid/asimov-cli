The whole flow of the program is the following:

We have the my_job.yaml file that contains the parameters such as the huggingface_token, model, dataset, dataset_subset, gpu_name, etc.

When we run main.py with the argument --config my_job.yaml then it uses the my_job.yaml parameters to initiate the process.

The main.py does the following in order:

1. It sets the environment variables with the --config my_job.yaml.

2. It goes to vast_manager.py and runs it with initiating the request for the GPU that has been set in the Env variables through my_job.yaml. It uses vastai cli to do it 
(read vast_docs.md on how the cli works in detail).

3. Then the request for GPU is initiated and the GPU send the signal that it's SSH has been recieved.

## Things that need to be done by AI (if you're reading this)

The CLI would allow users to add their Hugging Face Token so they can download the models & datasets and can publish the models to the Hugging Face repo.

Once they have set their token, it would be saved and would not be asked again unless on one exception (listed below in Custom Commands)

They would be abe to add the name of the model they want to finetune from HF. They can add the the dataset and subset (if applied) name from HF (Hugging Face).

They would be able to specify the name of their HF repo they want their fine-tuned model to be saved.

They would be able to specify the name of the GPU they want their model to be finetuned and the number of GPUs.

Basically this would allow then indirectly to edit the values set in my_job.yaml EXCEPT the docker_image value.


## Custom Commands

These are the custom commands through which the CLI would be opened in the terminal.

Activation Command: asimov # this command would open the CLI and would prompt the user to add the values mentioned above.
Token Edit: asimov token # This is the ONLY exception on which the user can change his Hugging Face token that was saved before.
Destroy Command: asimov destroy # This command would destroy the running instance.

## Design Principles

Use world class CLI designs. The dummy structure is given in /CLI files. Feel free to add your own UI and UX touch to it.

Your job is to make it real that can set the parameters mentioned above & tracks the progress from instance and shows the pregress bar.



