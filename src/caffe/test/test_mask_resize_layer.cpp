// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mask_resize_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;

template <typename TypeParam>
class MaskResizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MaskResizeLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 1, 21, 21)),
        blob_top_data_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~MaskResizeLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MaskResizeLayerTest, TestDtypesGPU);

TYPED_TEST(MaskResizeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MaskResizeParameter* mask_resize_param =
      layer_param.mutable_mask_resize_param();
  mask_resize_param->set_output_height(14);
  mask_resize_param->set_output_width(14);
  MaskResizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
