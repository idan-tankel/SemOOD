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
