# Usage:
## 1、准备模型
下载官方BitNet-b1.58-2B-4T模型,保存到00_external/models/目录  
使用转换脚本生成辅助模型文件
> python3 repack_gguf_for_v37.py ggml-model-i2_s.gguf ggml-model-i2_s_new.v37p
        
## 2、上传文件
将三个目录上传到KV260的目录下

## 3、固件烧录
> cd firmware  
> sudo ./init_fpga.sh

## 4、运行与测试方法
```bash
ssh ubuntu@192.168.31.116
cd /home/ubuntu/07_aligned_fused_engine
sudo sysctl -w kernel.sched_rt_runtime_us=-1
sudo chrt -f 99 taskset -c 0-3 ./semantic_engine /home/ubuntu/00_external/models/ggml-model-i2_s.gguf \
    -w /home/ubuntu/test_scheme13_v38/00_external/models/ggml-model-i2_s.v37p \
    -p "The hardware accelerator is" \
    -n 30 -t 4 -r 1.10
```
n:输出Token长度  
r：重复惩罚值，不建议修改  
t：4，使用kv260上4个A53核心  


## 提示词"Machine learning is~" 测试输出：

    OUTPUT: \define\[\] a method of data analysis that automates the prediction of outcomes based on historical data and patterns. It has been used in various fields such as medicine, finance, and marketing to improve decision making and predict future trends.
    
    In this section, we will explore how machine learning can be applied to solve real-life problems. We will discuss several case studies where machine learning has been successfully implemented to address complex challenges.
    
    Case Study 1: Predicting Heart Attack
    
    A hospital in New York
    
    ==== Performance Report ====
    Mode: V37 FPGA
    Prefill:  4 tokens in 357.18 ms (11.20 TPS)
    Decode:   100 tokens in 10.092 sec (100.92 ms/token)
    Speed:    9.91 TPS (Pure Decode), 9.57 TPS (End-to-End)
    Forward:  avg 100.90 ms/token (9.91 TPS)
