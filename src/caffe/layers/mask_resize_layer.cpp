// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include "caffe/layers/mask_resize_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  MaskResizeParameter mask_resize_param = this->layer_param_.mask_resize_param();
  CHECK_GT(mask_resize_param.output_height(), 0);
  CHECK_GT(mask_resize_param.output_width(), 0);
  output_width_ = mask_resize_param.output_width();
  output_height_ = mask_resize_param.output_height();
}

template <typename Dtype>
void MaskResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  input_channels_ = bottom[0]->channels();
  output_channels_ = input_channels_;
  input_height_ = bottom[0]->height();
  input_width_ = bottom[0]->width();
  MaskResizeParameter mask_resize_param = this->layer_param_.mask_resize_param();
  float factor_height = mask_resize_param.factor_height();
  float factor_width = mask_resize_param.factor_width();
  if(mask_resize_param.factor_height() > 0){
    output_height_ = int(bottom[0]->height() * factor_height);
  }
  if(mask_resize_param.factor_width() > 0){
    output_width_ = int(bottom[0]->width() * factor_width);
  }
  top[0]->Reshape(bottom[0]->num(), output_channels_, output_height_, output_width_);
}

template <typename Dtype>
void MaskResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MaskResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MaskResizeLayer);
#endif

INSTANTIATE_CLASS(MaskResizeLayer);
REGISTER_LAYER_CLASS(MaskResize);

}  // namespace caffe
