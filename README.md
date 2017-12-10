# Denoise-CNN-for-realtime-path-tracing
A convolutional neural network for denoising incomplete path tracing renders using Feature Pyrimad Networks

We experiment with the Feature Pyrimad Network of Lin et al. (https://arxiv.org/pdf/1612.03144.pdf) to perform denoising of low-sample-rate EXR renders. In conjunction with our CUDA path tracer (https://github.com/trevor-m/cuda-pathtrace) we somewhat manage to achieve realtime path tracing on a GTX 1080 ti with the big caveat that the quality of the result isn't good enough to be used in something like film production.

