# SemOOD
## overview
    The project currently contains an evaluation code for 2 datasets containing hard examples for Vision language models (VLM's)
    The 2 known benchmarks are                                                                                 
- *sugar-crepe* (available at [here](https://github.com/RAIVNLab/sugar-crepe)
 
- *MMBenchmark* (available at [OpenCompass Project page](https://opencompass.org.cn/mmbench))
## Citations


"...the **go to** statement should be abolished..." [[1]](#1).

## References
<a id="1">[1]</a> 



## experiments


according to a few executions, the prompt
```pyhton
"Question: The following is a multiple choice question. Choose an answer by it's number
     ...: \n 1.There is a tower in the image\n 2. There is a castle in the image.\n Answer:"
```
returns always A.

https://discuss.huggingface.co/t/trying-to-understand-system-prompts-with-llama-2-and-transformers-interface/59016


### further notes issues and FAQs


For installation and running of LLaMA2 using `transformers` python package (by  Huggingface.co) you will need to create a read token for the LLaMA resource. More information can be found here

[blogpost][llama2blog]

[llama2blog]: https://huggingface.co/blog/llama2#using-transformers