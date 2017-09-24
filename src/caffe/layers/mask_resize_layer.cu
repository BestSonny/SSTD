// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include "caffe/layers/mask_resize_layer.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype bilinear_interpolate(const Dtype* bottom_data, const int input_height, const int input_width, Dtype inverse_y, Dtype inverse_x) {

  // deal with cases that inverse elements are out of feature map boundary
  if (inverse_y < -0.5 || inverse_y > input_height - 0.5 || inverse_x < -0.5 || inverse_x > input_width - 0.5) {
    return 0.0;
  }

  if (inverse_y <= 0) inverse_y = 0;
  if (inverse_x <= 0) inverse_x = 0;

  int h_low = (int) inverse_y;
  int w_low = (int) inverse_x;
  int h_high;
  int w_high;

  if (h_low >= input_height - 1) {
    h_high = h_low = input_height - 1;
    inverse_y = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= input_width - 1) {
    w_high = w_low = input_width - 1;
    inverse_x = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }

  Dtype lh = inverse_y - h_low;
  Dtype lw = inverse_x - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;
  // do bilinear interpolation
  Dtype v1 = bottom_data[h_low * input_width + w_low];
  Dtype v2 = bottom_data[h_low * input_width + w_high];
  Dtype v3 = bottom_data[h_high * input_width + w_low];
  Dtype v4 = bottom_data[h_high * input_width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename Dtype>
__global__ void MaskResizeForward(const int nthreads, const Dtype* bottom_data, const int output_width, const int output_height, const int output_channels, const int input_width, const int input_height, const int input_channels, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in output mask
    int w = index % output_width;
    int h = (index / output_width) % output_height;
    int c = (index / output_width / output_height) % output_channels;
    int n = index / output_width / output_height / output_channels;
    Dtype ratio_h = static_cast<Dtype>(input_height) / static_cast<Dtype>(output_height);
    Dtype ratio_w = static_cast<Dtype>(input_width) / static_cast<Dtype>(output_width);
    Dtype inverse_x = w * ratio_w;
    Dtype inverse_y = h * ratio_h;

    const Dtype* offset_bottom_data = bottom_data + (n * input_channels + c) * input_height * input_width;

    top_data[index] = bilinear_interpolate(offset_bottom_data, input_height, input_width, inverse_y, inverse_x);
  }
}

template <typename Dtype>
void MaskResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  MaskResizeForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (count, bottom_data, output_width_, output_height_, output_channels_, input_width_, input_height_, input_channels_, top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__device__ Dtype getGradientWeight(Dtype argmax_h, Dtype argmax_w, const int h, const int w, const int height, const int width)
{
  if (argmax_h < -0.5 || argmax_h >(height - 0.5) || argmax_w < -0.5 || argmax_w >(width - 0.5))
    {
      return 0;
    }

  if (argmax_h < 0) argmax_h = 0;
  if (argmax_w < 0) argmax_w = 0;

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (Dtype)argmax_h_low;
  }
  else
    argmax_h_high = argmax_h_low + 1;

  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (Dtype)argmax_w_low;
  }
  else
    argmax_w_high = argmax_w_low + 1;

  Dtype weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  }
  else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}

template <typename Dtype>
__global__ void MaskResizeBackward(const int nthreads, const Dtype* top_diff, const int output_width, const int output_height, const int output_channels, const int input_width, const int input_height, const int input_channels, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in input mask
    int w = index % input_width;
    int h = (index / input_width) % input_height;
    int c = (index / input_width / input_height) % input_channels;
    int n = index / input_width / input_height / input_channels;

    Dtype gradient = 0.0;

    Dtype ratio_h = static_cast<Dtype>(input_height) / static_cast<Dtype>(output_height);
    Dtype ratio_w = static_cast<Dtype>(input_width)/ static_cast<Dtype>(output_width);
    Dtype map_x = static_cast<Dtype>(w) / ratio_w;
    Dtype map_y = static_cast<Dtype>(h) / ratio_h;

    int output_h_start = floor(map_y);
    int output_w_start = floor(map_x);
    int output_h_end = output_h_start + 1;
    int output_w_end = output_w_start + 1;

    int offset = (n * output_channels + c) * output_height * output_width;
    const Dtype* offset_top_diff = top_diff + offset;
    for (int ph = output_h_start; ph <= output_h_end; ++ph) {
      for (int pw = output_w_start; pw <= output_w_end; ++pw) {
        // map the output index back to feature map index
        Dtype iw = static_cast<Dtype>(pw) * ratio_w;
        Dtype ih = static_cast<Dtype>(ph) * ratio_h;
        // check whether bottom element of this index will affect output element
        if (fabs(iw - w) >= 1 || fabs(ih - h) >= 1) {
          continue;
        }
        Dtype weight = getGradientWeight(ih, iw, h, w, input_height, input_width);
        gradient += weight * offset_top_diff[ph * output_width + pw];
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void MaskResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();

  MaskResizeBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (count, top_diff, output_width_, output_height_, output_channels_, input_width_, input_height_, input_channels_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskResizeLayer);

}
