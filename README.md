# PAL*M: Property Attestation for Large Generative Models

## Environment Setup

PAL*M relies on Intel TDX and NVIDIA H100 CC mode. Please follow the official NVIDIA Secure AI deployment guide to correctly set up the environment:

https://docs.nvidia.com/cc-deployment-guide-tdx.pdf

Make sure the Trusted Domain (TD) is properly configured according to the deployment guide.

### Python dependencies
Once inside the TD, install the required Python dependencies for running PAL*M and reproducing the experiments:
```
pip install -r requirements.txt
```

## Running the experiements
To run property attestation experiments, note that sudo access is required if quote generation or GPU attestation is used. 

Example of running the proof of finetuning with Llama 3.1 8B model:
```
python3 main_LLM.py --attestation_type finetune --model llama --measure
```

Example of running the proof of training: 
```
python3 main_LLM.py --attestation_type pretrain --measure
```

The result of the time measurement will be written to llm_result.csv. The output generated from the operation, including the evidence will be stored in the llm_output directory

We also provide a script to run all experiments 5 times each, which were used in the paper. This script will output a raw time (in second) for each measurement steps
```
./run_everything.sh
```