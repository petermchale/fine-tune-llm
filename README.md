## Parameter-efficient fine-tuning an open-source 7B-parameter LLM for instruction following 

I fine-tuned a [7B LLM](https://huggingface.co/Salesforce/xgen-7b-8k-base) on a public [instruction dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using LoRA for 0.5 epochs over 5 hours on a single-GPU `g4dn.xlarge` EC2 instance. 

Due to the size of the LLM, I had to increase the volume size of the EC2 instance: 
```
(pytorch) du -hs .cache/huggingface/hub/models--Salesforce--xgen-7b-8k-base
26G     .cache/huggingface/hub/models--Salesforce--xgen-7b-8k-base
```

The training code uses the torch, transformers, peft and trl libraries. See `requirements.txt`.

The training run is tracked [here](https://wandb.ai/peter-thomas-mchale/huggingface/runs/8gdlnzsd?workspace=user-peter-thomas-mchale) and the fine-tuned model adaptors are available [here](https://huggingface.co/petermchale/xgen-7b-tuned-alpaca).

## Credit

https://www.youtube.com/watch?v=JNMVulH7fCo



