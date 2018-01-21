# Denoise-CNN-for-realtime-path-tracing

A convolutional neural network for denoising incomplete path tracing renders using Feature Pyramid Networks

We experiment with the Feature Pyramid Network of Lin et al. (https://arxiv.org/pdf/1612.03144.pdf) to perform denoising of low-sample-rate EXR renders. In conjunction with our CUDA path tracer (https://github.com/trevor-m/cuda-pathtrace) we somewhat manage to achieve realtime path tracing on a GTX 1080 ti with the big caveat that the quality of the result isn't good enough to be used in something like film production.

![alt text](https://github.com/ShayanPersonal/Denoise-CNN-for-realtime-path-tracing/blob/master/results/12553296/in74000.png?raw=true)

![alt text](https://github.com/ShayanPersonal/Denoise-CNN-for-realtime-path-tracing/blob/master/results/12553296/out74000.png?raw=true)
