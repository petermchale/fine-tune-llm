## Fine-tune an open-source 7B-parameter LLM for instruction following 

I trained a [7B LLM](https://huggingface.co/Salesforce/xgen-7b-8k-base) on a public [instruction dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using LoRA for 1 epoch over a period of about 2 days on a single-GPU `g4dn.xlarge` EC2 instance. 

Due to the size of the LLM, I had to increase the volume size of the EC2 instance: 
```
(pytorch) du -hs .cache/huggingface/hub/models--Salesforce--xgen-7b-8k-base
26G     .cache/huggingface/hub/models--Salesforce--xgen-7b-8k-base
```

The training code uses the torch, transformers, peft and trl libraries. See `requirements.txt`.

## Credit

https://www.youtube.com/watch?v=JNMVulH7fCo



